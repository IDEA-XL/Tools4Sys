import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime

import torch
import torch.distributed as dist
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed

from progen2.checkpoint import PROGEN2_SGRPO_VARIANT, stamp_checkpoint_variant
from progen2.data.prompts import load_prompt_texts
from progen2.modeling.wrapper import OfficialProGen2CausalLM
from progen2.rewards import CompositeProteinReward, compute_group_diversity_reward
from progen2.rl.policy import ProGen2Policy
from rl_shared.sgrpo import compute_clipped_grpo_loss, compute_grouped_advantages, compute_sgrpo_advantages


logger = logging.getLogger(__name__)


@dataclass
class ProGen2TrainConfig:
    model_variant: str
    official_code_dir: str
    tokenizer_path: str
    init_checkpoint_dir: str
    ref_checkpoint_dir: str | None = None
    checkpoint_subdir: str | None = None
    output_dir: str | None = None
    overwrite_output_dir: bool = False
    seed: int = 42
    prompt_path: str = ''
    per_device_prompt_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    bf16: bool = False
    max_steps: int = 50
    learning_rate: float = 1e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    beta: float = 0.01
    epsilon: float = 0.2
    logging_steps: int = 1
    save_steps: int = 25
    save_total_limit: int = 3
    report_to: list[str] = field(default_factory=list)
    rl_algorithm: str = 'coupled_sgrpo'
    num_generations: int = 4
    supergroup_num_groups: int = 2
    group_advantage_weight: float = 0.5
    max_new_tokens: int = 128
    top_p: float = 0.95
    temperature: float = 0.8
    reward_calibration_size: int = 4096
    rewards: dict = field(default_factory=dict)


def load_config(path):
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    config = ProGen2TrainConfig(**raw)
    if config.model_variant != PROGEN2_SGRPO_VARIANT:
        raise ValueError(
            f'Expected model_variant={PROGEN2_SGRPO_VARIANT!r}, got {config.model_variant!r}'
        )
    if config.ref_checkpoint_dir is None:
        config.ref_checkpoint_dir = config.init_checkpoint_dir
    if not config.prompt_path:
        raise ValueError('prompt_path is required')
    if config.per_device_prompt_batch_size <= 0:
        raise ValueError('per_device_prompt_batch_size must be positive')
    if config.gradient_accumulation_steps <= 0:
        raise ValueError('gradient_accumulation_steps must be positive')
    if config.max_steps <= 0:
        raise ValueError('max_steps must be positive')
    if config.num_generations <= 1:
        raise ValueError('num_generations must be greater than 1')
    if config.rl_algorithm not in {'coupled_grpo', 'coupled_sgrpo'}:
        raise ValueError("rl_algorithm must be 'coupled_grpo' or 'coupled_sgrpo'")
    if config.rl_algorithm == 'coupled_sgrpo' and config.supergroup_num_groups <= 1:
        raise ValueError('supergroup_num_groups must be greater than 1 for coupled_sgrpo')
    if not 0.0 <= config.group_advantage_weight <= 1.0:
        raise ValueError('group_advantage_weight must be in [0, 1]')
    if not config.rewards:
        raise ValueError('rewards config is required')
    return config


def resolve_output_dir(config, config_path):
    if config.output_dir is not None:
        return config.output_dir
    cluster_root = '/public/home/xinwuye/ai4s-tool-joint-train'
    if os.path.isdir(cluster_root):
        base_dir = os.path.join(cluster_root, 'runs', 'progen2_sgrpo')
    else:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
        base_dir = os.path.join(repo_root, 'runs', 'progen2_sgrpo')
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(base_dir, f'{config_name}_{timestamp}')


def _cycle_prompt_batch(prompts, batch_size, start_index):
    batch = []
    for offset in range(batch_size):
        batch.append(prompts[(start_index + offset) % len(prompts)])
    return batch


def _write_jsonl(path, payload):
    with open(path, 'a') as handle:
        handle.write(json.dumps(payload, sort_keys=True) + '\n')


class ProGen2SGRPOTrainer:
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        self.accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with=config.report_to if config.report_to else None,
            mixed_precision='bf16' if config.bf16 else 'no',
        )
        self.device = self.accelerator.device
        set_seed(config.seed, device_specific=True)

        self.prompts = load_prompt_texts(config.prompt_path)
        self._prompt_cursor = self.accelerator.process_index * config.per_device_prompt_batch_size
        self.num_return_sequences = (
            config.num_generations * config.supergroup_num_groups
            if config.rl_algorithm == 'coupled_sgrpo'
            else config.num_generations
        )

        self.policy = ProGen2Policy(
            OfficialProGen2CausalLM(
                official_code_dir=config.official_code_dir,
                checkpoint_dir=config.init_checkpoint_dir,
                tokenizer_path=config.tokenizer_path,
                checkpoint_subdir=config.checkpoint_subdir,
                device=self.device,
                use_fp16=not config.bf16,
            ),
            trainable=True,
        )
        self.reference = ProGen2Policy(
            OfficialProGen2CausalLM(
                official_code_dir=config.official_code_dir,
                checkpoint_dir=config.ref_checkpoint_dir,
                tokenizer_path=config.tokenizer_path,
                checkpoint_subdir=config.checkpoint_subdir,
                device=self.device,
                use_fp16=not config.bf16,
            ),
            trainable=False,
        )
        optimizer = torch.optim.AdamW(
            self.policy.model.trainable_parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        self.policy.model.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.policy.model.model,
            optimizer,
            scheduler,
        )
        self.reward_model = CompositeProteinReward(config.rewards, device=self.device)
        self.metrics_path = os.path.join(output_dir, 'metrics.jsonl')
        self.state_path = os.path.join(output_dir, 'trainer_state.json')
        self.global_step = 0

        if config.report_to:
            init_kwargs = {}
            if 'wandb' in config.report_to:
                init_kwargs['wandb'] = {'name': os.path.basename(output_dir)}
            self.accelerator.init_trackers('progen2-sgrpo', config=asdict(config), init_kwargs=init_kwargs or None)

    def _next_prompt_batch(self):
        batch = _cycle_prompt_batch(self.prompts, self.config.per_device_prompt_batch_size, self._prompt_cursor)
        self._prompt_cursor = (self._prompt_cursor + self.config.per_device_prompt_batch_size) % len(self.prompts)
        return batch

    def _calibration_sequences(self):
        remaining = int(self.config.reward_calibration_size)
        collected = []
        while remaining > 0:
            prompts = _cycle_prompt_batch(self.prompts, min(len(self.prompts), remaining), len(collected))
            rollout = self.policy.generate_rollouts(
                prompts,
                num_return_sequences=1,
                max_new_tokens=self.config.max_new_tokens,
                top_p=self.config.top_p,
                temperature=self.config.temperature,
                seed=self.config.seed + len(collected),
            )
            valid = [sequence for sequence in rollout.protein_sequences if sequence]
            collected.extend(valid)
            remaining = int(self.config.reward_calibration_size) - len(collected)
        return collected[: self.config.reward_calibration_size]

    def calibrate(self):
        calibration = None
        if self.accelerator.is_main_process:
            calibration_sequences = self._calibration_sequences()
            calibration = self.reward_model.calibrate(calibration_sequences)
        payload = [calibration]
        if self.accelerator.num_processes > 1 and dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(payload, src=0)
        self.reward_model.calibration = payload[0]

    def _score_rollout_rewards(self, sequences):
        valid_indices = [idx for idx, sequence in enumerate(sequences) if sequence]
        rewards = [0.0] * len(sequences)
        metrics = {'invalid_sequence_rate': 1.0 if not valid_indices else 1.0 - (len(valid_indices) / len(sequences))}
        if valid_indices:
            valid_sequences = [sequences[idx] for idx in valid_indices]
            valid_rewards, reward_metrics = self.reward_model.score(valid_sequences)
            for target_idx, reward in zip(valid_indices, valid_rewards):
                rewards[target_idx] = float(reward)
            metrics.update(reward_metrics)
        return torch.tensor(rewards, device=self.device, dtype=torch.float32), metrics

    def _score_group_rewards(self, sequences):
        group_rewards = []
        for start in range(0, len(sequences), self.config.num_generations):
            group_sequences = [sequence for sequence in sequences[start:start + self.config.num_generations] if sequence]
            if len(group_sequences) < 2:
                group_rewards.append(0.0)
            else:
                group_rewards.append(float(compute_group_diversity_reward(group_sequences)))
        return torch.tensor(group_rewards, device=self.device, dtype=torch.float32)

    def _compute_advantages(self, rollout_rewards, group_rewards):
        advantages = []
        expanded_group_advantages = []
        rollout_advantages = []
        metrics = {}
        chunk_rollout = self.num_return_sequences
        chunk_groups = (
            self.config.supergroup_num_groups
            if self.config.rl_algorithm == 'coupled_sgrpo'
            else 1
        )
        for prompt_idx in range(self.config.per_device_prompt_batch_size):
            start = prompt_idx * chunk_rollout
            end = start + chunk_rollout
            prompt_rollout_rewards = rollout_rewards[start:end]
            group_start = prompt_idx * chunk_groups
            group_end = group_start + chunk_groups
            prompt_group_rewards = group_rewards[group_start:group_end]
            if self.config.rl_algorithm == 'coupled_sgrpo':
                prompt_adv, prompt_group_adv, prompt_rollout_adv, prompt_metrics = compute_sgrpo_advantages(
                    rollout_rewards=prompt_rollout_rewards,
                    group_rewards=prompt_group_rewards,
                    num_generations=self.config.num_generations,
                    supergroup_num_groups=self.config.supergroup_num_groups,
                    group_advantage_weight=self.config.group_advantage_weight,
                    scale_rewards=False,
                )
            else:
                prompt_adv, _, zero_std_ratio = compute_grouped_advantages(
                    rewards=prompt_rollout_rewards,
                    num_generations=self.config.num_generations,
                    scale_rewards=False,
                )
                prompt_rollout_adv = prompt_adv
                prompt_group_adv = torch.zeros_like(prompt_rollout_rewards)
                prompt_metrics = {
                    'rollout_zero_std_ratio': zero_std_ratio,
                    'group_zero_std_ratio': float('nan'),
                    'group_reward_mean': prompt_group_rewards.mean().item() if prompt_group_rewards.numel() else float('nan'),
                }
            advantages.append(prompt_adv)
            expanded_group_advantages.append(prompt_group_adv)
            rollout_advantages.append(prompt_rollout_adv)
            for key, value in prompt_metrics.items():
                metrics.setdefault(key, []).append(float(value))
        merged_metrics = {key: sum(values) / len(values) for key, values in metrics.items()}
        return (
            torch.cat(advantages, dim=0),
            torch.cat(expanded_group_advantages, dim=0),
            torch.cat(rollout_advantages, dim=0),
            merged_metrics,
        )

    def save_checkpoint(self):
        checkpoint_dir = os.path.join(self.output_dir, f'checkpoint-{self.global_step:06d}')
        unwrapped = self.accelerator.unwrap_model(self.policy.model.model)
        unwrapped.save_pretrained(checkpoint_dir, state_dict=self.accelerator.get_state_dict(self.policy.model.model))
        trainer_state = {
            'global_step': int(self.global_step),
            'config': asdict(self.config),
        }
        stamp_checkpoint_variant(trainer_state, PROGEN2_SGRPO_VARIANT)
        torch.save(trainer_state, os.path.join(checkpoint_dir, 'trainer_state.pt'))

    def _log(self, metrics):
        payload = {'step': int(self.global_step), **metrics}
        if self.accelerator.is_main_process:
            _write_jsonl(self.metrics_path, payload)
            with open(self.state_path, 'w') as handle:
                json.dump({'global_step': self.global_step}, handle, indent=2, sort_keys=True)
        if self.config.report_to:
            self.accelerator.log(metrics, step=self.global_step)

    def train(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.calibrate()

        for step_idx in range(self.config.max_steps):
            prompts = self._next_prompt_batch()
            rollout = self.policy.generate_rollouts(
                prompts,
                num_return_sequences=self.num_return_sequences,
                max_new_tokens=self.config.max_new_tokens,
                top_p=self.config.top_p,
                temperature=self.config.temperature,
                seed=self.config.seed + step_idx,
            )
            rollout_rewards, reward_metrics = self._score_rollout_rewards(rollout.protein_sequences)
            group_rewards = self._score_group_rewards(rollout.protein_sequences)
            advantages, group_advantages, rollout_advantages, advantage_metrics = self._compute_advantages(
                rollout_rewards,
                group_rewards,
            )

            with torch.no_grad():
                old_log_probs, completion_mask = self.policy.per_token_logps(
                    rollout.full_token_ids,
                    rollout.full_attention_mask,
                    rollout.generated_mask,
                    requires_grad=False,
                )
                ref_log_probs, _ = self.reference.per_token_logps(
                    rollout.full_token_ids,
                    rollout.full_attention_mask,
                    rollout.generated_mask,
                    requires_grad=False,
                )

            new_log_probs, completion_mask = self.policy.per_token_logps(
                rollout.full_token_ids,
                rollout.full_attention_mask,
                rollout.generated_mask,
                requires_grad=True,
            )
            loss, loss_metrics = compute_clipped_grpo_loss(
                new_log_probs=new_log_probs,
                old_log_probs=old_log_probs,
                advantages=advantages,
                completion_mask=completion_mask,
                epsilon=self.config.epsilon,
                ref_log_probs=ref_log_probs,
                beta=self.config.beta,
            )

            self.optimizer.zero_grad(set_to_none=True)
            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(self.policy.model.trainable_parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            self.global_step += 1
            metrics = {
                'loss': float(loss.detach().item()),
                'reward_mean': float(rollout_rewards.mean().item()),
                'group_reward_mean': float(group_rewards.mean().item()),
                'advantage_mean': float(advantages.mean().item()),
                'group_advantage_mean': float(group_advantages.mean().item()),
                'rollout_advantage_mean': float(rollout_advantages.mean().item()),
                **{key: float(value) for key, value in reward_metrics.items()},
                **{key: float(value) for key, value in advantage_metrics.items()},
                **{key: float(value.item() if hasattr(value, 'item') else value) for key, value in loss_metrics.items()},
            }
            if self.global_step % self.config.logging_steps == 0:
                self._log(metrics)
            if self.global_step % self.config.save_steps == 0:
                self.save_checkpoint()

        if self.config.report_to:
            self.accelerator.end_training()
