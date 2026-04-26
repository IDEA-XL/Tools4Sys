import json
import logging
import os
import random
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
from progen2.rewards import (
    CompositeProteinReward,
    compute_group_diversity_loo_credits,
    compute_group_diversity_reward_or_zero,
)
from progen2.rewards.composite import (
    REWARD_NAME_ORDER,
    normalize_protein_reward_weights,
    normalize_reward_compute_every_n_steps,
)
from progen2.rl.policy import ProGen2Policy, ProGen2RolloutBatch
from rl_shared.sampling import normalize_scalar_or_range, sample_scalar_or_range
from rl_shared.sgrpo import (
    VALID_GROUP_REWRAD_CREDITS,
    VALID_SGRPO_HIERARCHIES,
    compute_clipped_grpo_loss,
    compute_grouped_advantages,
    compute_sgrpo_advantages,
    validate_reward_threshold_names,
)


logger = logging.getLogger(__name__)


def default_reward_compute_every_n_steps():
    return normalize_reward_compute_every_n_steps(None)


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
    rl_algorithm: str = 'sgrpo'
    num_generations: int = 4
    supergroup_num_groups: int = 2
    group_advantage_weight: float = 0.5
    hierarchy: str = 'advantage_sum'
    naturalness: float | None = None
    foldability: float | None = None
    stability: float | None = None
    developability: float | None = None
    individual_reward_thresholds: dict[str, float | None] = field(default_factory=dict)
    group_rewrad_credit: str = 'broadcast'
    group_rewrad_credit_temperature: float = 1.0
    reward_compute_every_n_steps: dict[str, int] = field(
        default_factory=default_reward_compute_every_n_steps
    )
    max_new_tokens: int = 128
    top_p: float = 0.95
    temperature: float | list[float] = 0.8
    reward_calibration_size: int = 4096
    reward_calibration_prompt_batch_size: int = 8
    rewards: dict = field(default_factory=dict)


def load_config(path):
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    int_fields = [
        'seed',
        'per_device_prompt_batch_size',
        'gradient_accumulation_steps',
        'max_steps',
        'logging_steps',
        'save_steps',
        'save_total_limit',
        'num_generations',
        'supergroup_num_groups',
        'max_new_tokens',
        'reward_calibration_size',
        'reward_calibration_prompt_batch_size',
    ]
    float_fields = [
        'learning_rate',
        'adam_beta1',
        'adam_beta2',
        'adam_eps',
        'weight_decay',
        'max_grad_norm',
        'beta',
        'epsilon',
        'group_advantage_weight',
        'top_p',
    ]
    for field_name in int_fields:
        if field_name in raw and raw[field_name] is not None:
            raw[field_name] = int(raw[field_name])
    for field_name in float_fields:
        if field_name in raw and raw[field_name] is not None:
            raw[field_name] = float(raw[field_name])
    if 'group_rewrad_credit_temperature' in raw and raw['group_rewrad_credit_temperature'] is not None:
        raw['group_rewrad_credit_temperature'] = float(raw['group_rewrad_credit_temperature'])
    raw['reward_compute_every_n_steps'] = normalize_reward_compute_every_n_steps(
        raw.get('reward_compute_every_n_steps')
    )
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
    if config.rl_algorithm not in {'grpo', 'sgrpo'}:
        raise ValueError("rl_algorithm must be 'grpo' or 'sgrpo'")
    if config.supergroup_num_groups <= 0:
        raise ValueError('supergroup_num_groups must be positive')
    if config.rl_algorithm == 'sgrpo' and config.supergroup_num_groups <= 1:
        raise ValueError('supergroup_num_groups must be greater than 1 for sgrpo')
    if not 0.0 <= config.group_advantage_weight <= 1.0:
        raise ValueError('group_advantage_weight must be in [0, 1]')
    config.rollout_reward_weights = normalize_protein_reward_weights(
        {
            'naturalness': config.naturalness,
            'foldability': config.foldability,
            'stability': config.stability,
            'developability': config.developability,
        }
    )
    config.temperature = normalize_scalar_or_range(
        config.temperature,
        name='temperature',
        min_exclusive=0.0,
    )
    config.individual_reward_thresholds = validate_reward_threshold_names(
        config.individual_reward_thresholds,
        REWARD_NAME_ORDER,
    )
    _validate_active_protein_reward_thresholds(
        config.individual_reward_thresholds,
        config.rollout_reward_weights,
    )
    if config.hierarchy not in VALID_SGRPO_HIERARCHIES:
        raise ValueError(
            f"hierarchy must be one of {sorted(VALID_SGRPO_HIERARCHIES)}, got {config.hierarchy!r}"
        )
    if config.group_rewrad_credit not in VALID_GROUP_REWRAD_CREDITS:
        raise ValueError(
            f"group_rewrad_credit must be one of {sorted(VALID_GROUP_REWRAD_CREDITS)}, "
            f'got {config.group_rewrad_credit!r}'
        )
    if config.group_rewrad_credit_temperature <= 0.0:
        raise ValueError('group_rewrad_credit_temperature must be positive')
    has_active_threshold = any(
        threshold is not None for threshold in config.individual_reward_thresholds.values()
    )
    if config.rl_algorithm != 'sgrpo':
        if has_active_threshold:
            raise ValueError('individual_reward_thresholds is only supported when rl_algorithm=sgrpo')
        if config.hierarchy != 'advantage_sum':
            raise ValueError('hierarchy is only supported when rl_algorithm=sgrpo')
        if config.group_rewrad_credit != 'broadcast':
            raise ValueError('group_rewrad_credit is only supported when rl_algorithm=sgrpo')
        if config.group_rewrad_credit_temperature != 1.0:
            raise ValueError('group_rewrad_credit_temperature is only supported when rl_algorithm=sgrpo')
    for reward_name, threshold in config.individual_reward_thresholds.items():
        if threshold is not None and config.reward_compute_every_n_steps[reward_name] != 1:
            raise ValueError(
                'thresholded progen2 rewards must be computed every step; '
                f'{reward_name!r} has every_n_steps={config.reward_compute_every_n_steps[reward_name]}'
            )
    if not config.rewards:
        raise ValueError('rewards config is required')
    if config.reward_calibration_prompt_batch_size <= 0:
        raise ValueError('reward_calibration_prompt_batch_size must be positive')
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


def _validate_active_protein_reward_thresholds(thresholds, reward_weights):
    for reward_name in REWARD_NAME_ORDER:
        threshold = thresholds.get(reward_name)
        if threshold is not None and reward_weights[reward_name] <= 0.0:
            raise ValueError(
                f'individual_reward_thresholds[{reward_name!r}] requires a positive rollout reward weight, '
                f'got {reward_weights[reward_name]}'
            )


def _cycle_prompt_batch(prompts, batch_size, start_index):
    batch = []
    for offset in range(batch_size):
        batch.append(prompts[(start_index + offset) % len(prompts)])
    return batch


def _write_jsonl(path, payload):
    with open(path, 'a') as handle:
        handle.write(json.dumps(payload, sort_keys=True) + '\n')


def default_reward_batch_size(config):
    return int(
        config.per_device_prompt_batch_size
        * config.num_generations
        * config.supergroup_num_groups
    )


def _merge_rollout_batches(batches, pad_token_id):
    if not batches:
        raise ValueError('cannot merge an empty rollout batch list')
    if len(batches) == 1:
        return batches[0]

    max_len = max(batch.full_token_ids.size(1) for batch in batches)
    padded_ids = []
    padded_attention = []
    padded_generated_mask = []
    for batch in batches:
        pad_len = max_len - batch.full_token_ids.size(1)
        if pad_len < 0:
            raise ValueError('internal error: negative rollout pad length')
        if pad_len == 0:
            padded_ids.append(batch.full_token_ids)
            padded_attention.append(batch.full_attention_mask)
            padded_generated_mask.append(batch.generated_mask)
            continue
        id_pad = torch.full(
            (batch.full_token_ids.size(0), pad_len),
            fill_value=pad_token_id,
            device=batch.full_token_ids.device,
            dtype=batch.full_token_ids.dtype,
        )
        zero_pad = torch.zeros(
            (batch.full_token_ids.size(0), pad_len),
            device=batch.full_token_ids.device,
            dtype=batch.full_attention_mask.dtype,
        )
        mask_pad = torch.zeros(
            (batch.full_token_ids.size(0), pad_len),
            device=batch.full_token_ids.device,
            dtype=batch.generated_mask.dtype,
        )
        padded_ids.append(torch.cat([batch.full_token_ids, id_pad], dim=1))
        padded_attention.append(torch.cat([batch.full_attention_mask, zero_pad], dim=1))
        padded_generated_mask.append(torch.cat([batch.generated_mask, mask_pad], dim=1))

    prompt_texts = []
    decoded_texts = []
    protein_sequences = []
    for batch in batches:
        prompt_texts.extend(batch.prompt_texts)
        decoded_texts.extend(batch.decoded_texts)
        protein_sequences.extend(batch.protein_sequences)

    return ProGen2RolloutBatch(
        prompt_texts=prompt_texts,
        prompt_lengths=torch.cat([batch.prompt_lengths for batch in batches], dim=0),
        full_token_ids=torch.cat(padded_ids, dim=0),
        full_attention_mask=torch.cat(padded_attention, dim=0),
        generated_mask=torch.cat(padded_generated_mask, dim=0),
        decoded_texts=decoded_texts,
        protein_sequences=protein_sequences,
    )


class ProGen2SGRPOTrainer:
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        self.autocast_dtype = torch.bfloat16 if config.bf16 else None
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
            if config.rl_algorithm == 'sgrpo'
            else config.num_generations
        )

        self.policy = ProGen2Policy(
            OfficialProGen2CausalLM(
                official_code_dir=config.official_code_dir,
                checkpoint_dir=config.init_checkpoint_dir,
                tokenizer_path=config.tokenizer_path,
                checkpoint_subdir=config.checkpoint_subdir,
                device=self.device,
                use_fp16=False,
                autocast_dtype=self.autocast_dtype,
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
                use_fp16=False,
                autocast_dtype=self.autocast_dtype,
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
        self.reward_model = CompositeProteinReward(
            config.rewards,
            device=self.device,
            default_reward_batch_size=default_reward_batch_size(config),
            reward_compute_every_n_steps=config.reward_compute_every_n_steps,
            reward_weights=config.rollout_reward_weights,
        )
        self.metrics_path = os.path.join(output_dir, 'metrics.jsonl')
        self.state_path = os.path.join(output_dir, 'trainer_state.json')
        self.global_step = 0
        self._cuda_run_max_reserved = 0
        self._cuda_run_max_allocated = 0
        self._cuda_phase_run_max_reserved = {
            'rollout': 0,
            'reward': 0,
            'training': 0,
        }
        self._cuda_phase_run_max_allocated = {
            'rollout': 0,
            'reward': 0,
            'training': 0,
        }

        if config.report_to:
            init_kwargs = {}
            if 'wandb' in config.report_to:
                init_kwargs['wandb'] = {'name': os.path.basename(output_dir)}
            self.accelerator.init_trackers('progen2-sgrpo', config=asdict(config), init_kwargs=init_kwargs or None)

    def _generate_rollouts(self, prompts, *, num_return_sequences, seed):
        rng = random.Random(seed)
        temperatures = [
            sample_scalar_or_range(
                self.config.temperature,
                rng,
                name='temperature',
                min_exclusive=0.0,
            )
            for _ in prompts
        ]
        if all(temperature == temperatures[0] for temperature in temperatures):
            return self.policy.generate_rollouts(
                prompts,
                num_return_sequences=num_return_sequences,
                max_new_tokens=self.config.max_new_tokens,
                top_p=self.config.top_p,
                temperature=temperatures[0],
                seed=seed,
            )

        batches = []
        for prompt_idx, (prompt, temperature) in enumerate(zip(prompts, temperatures)):
            batches.append(
                self.policy.generate_rollouts(
                    [prompt],
                    num_return_sequences=num_return_sequences,
                    max_new_tokens=self.config.max_new_tokens,
                    top_p=self.config.top_p,
                    temperature=temperature,
                    seed=seed + prompt_idx,
                )
            )
        return _merge_rollout_batches(
            batches,
            pad_token_id=self.policy.model.tokenizer.pad_token_id,
        )

    def _next_prompt_batch(self):
        batch = _cycle_prompt_batch(self.prompts, self.config.per_device_prompt_batch_size, self._prompt_cursor)
        self._prompt_cursor = (self._prompt_cursor + self.config.per_device_prompt_batch_size) % len(self.prompts)
        return batch

    def _calibration_sequences(self):
        remaining = int(self.config.reward_calibration_size)
        collected = []
        calibration_cursor = 0
        prompt_batch_size = int(self.config.reward_calibration_prompt_batch_size)
        logger.info(
            'Starting reward calibration: target_sequences=%d prompt_batch_size=%d',
            remaining,
            prompt_batch_size,
        )
        while remaining > 0:
            prompts = _cycle_prompt_batch(self.prompts, min(prompt_batch_size, remaining), calibration_cursor)
            calibration_cursor = (calibration_cursor + len(prompts)) % len(self.prompts)
            rollout = self._generate_rollouts(
                prompts,
                num_return_sequences=1,
                seed=self.config.seed + len(collected),
            )
            valid = [sequence for sequence in rollout.protein_sequences if sequence]
            collected.extend(valid)
            remaining = int(self.config.reward_calibration_size) - len(collected)
            logger.info(
                'Reward calibration progress: collected=%d remaining=%d',
                len(collected),
                max(remaining, 0),
            )
        return collected[: self.config.reward_calibration_size]

    def calibrate(self):
        calibration = None
        if self.accelerator.is_main_process:
            calibration_sequences = self._calibration_sequences()
            logger.info('Scoring reward calibration statistics on %d sequences', len(calibration_sequences))
            calibration = self.reward_model.calibrate(calibration_sequences)
            logger.info('Reward calibration complete')
        payload = [calibration]
        if self.accelerator.num_processes > 1 and dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(payload, src=0)
        self.reward_model.calibration = payload[0]

    def _score_rollout_rewards(self, sequences, *, step_number):
        valid_indices = [idx for idx, sequence in enumerate(sequences) if sequence]
        rewards = [0.0] * len(sequences)
        individual_reward_values = {
            reward_name: [0.0] * len(sequences) for reward_name in REWARD_NAME_ORDER
        }
        metrics = {'invalid_sequence_rate': 1.0 if not valid_indices else 1.0 - (len(valid_indices) / len(sequences))}
        if valid_indices:
            valid_sequences = [sequences[idx] for idx in valid_indices]
            reward_details, reward_metrics = self.reward_model.score_details(
                valid_sequences,
                step_number=step_number,
            )
            valid_rewards = reward_details['total']
            for target_idx, reward in zip(valid_indices, valid_rewards):
                rewards[target_idx] = float(reward)
            for reward_name in REWARD_NAME_ORDER:
                for target_idx, reward_value in zip(valid_indices, reward_details[reward_name]):
                    individual_reward_values[reward_name][target_idx] = float(reward_value)
            metrics.update(reward_metrics)
        individual_reward_tensors = {
            reward_name: torch.tensor(values, device=self.device, dtype=torch.float32)
            for reward_name, values in individual_reward_values.items()
        }
        return (
            torch.tensor(rewards, device=self.device, dtype=torch.float32),
            metrics,
            individual_reward_tensors,
        )

    def _score_group_rewards(self, sequences):
        group_rewards = []
        for start in range(0, len(sequences), self.config.num_generations):
            group_sequences = sequences[start:start + self.config.num_generations]
            group_rewards.append(float(compute_group_diversity_reward_or_zero(group_sequences)))
        return torch.tensor(group_rewards, device=self.device, dtype=torch.float32)

    def _score_group_reward_credits(self, sequences):
        group_credits = []
        for start in range(0, len(sequences), self.config.num_generations):
            group_sequences = sequences[start:start + self.config.num_generations]
            group_credits.extend(compute_group_diversity_loo_credits(group_sequences))
        return torch.tensor(group_credits, device=self.device, dtype=torch.float32)

    def _build_group_mean_individual_rewards(self, individual_reward_tensors):
        outputs = {}
        for reward_name, reward_values in individual_reward_tensors.items():
            if reward_values.dim() != 1:
                raise ValueError(f'individual reward tensor for {reward_name!r} must be 1D')
            if reward_values.numel() % self.config.num_generations != 0:
                raise ValueError(
                    'individual reward tensor length must be divisible by num_generations: '
                    f'{reward_name!r} has {reward_values.numel()} vs {self.config.num_generations}'
                )
            outputs[reward_name] = reward_values.view(-1, self.config.num_generations).mean(dim=1)
        return outputs

    def _compute_advantages(
        self,
        rollout_rewards,
        group_rewards,
        group_mean_individual_rewards=None,
        group_reward_credits=None,
    ):
        advantages = []
        expanded_group_advantages = []
        rollout_advantages = []
        metrics = {}
        chunk_rollout = self.num_return_sequences
        chunk_groups = (
            self.config.supergroup_num_groups
            if self.config.rl_algorithm == 'sgrpo'
            else 1
        )
        for prompt_idx in range(self.config.per_device_prompt_batch_size):
            start = prompt_idx * chunk_rollout
            end = start + chunk_rollout
            prompt_rollout_rewards = rollout_rewards[start:end]
            group_start = prompt_idx * chunk_groups
            group_end = group_start + chunk_groups
            prompt_group_rewards = group_rewards[group_start:group_end]
            prompt_group_mean_individual_rewards = None
            if group_mean_individual_rewards is not None:
                prompt_group_mean_individual_rewards = {
                    reward_name: reward_means[group_start:group_end]
                    for reward_name, reward_means in group_mean_individual_rewards.items()
                }
            if self.config.rl_algorithm == 'sgrpo':
                prompt_group_reward_credits = None
                if group_reward_credits is not None:
                    prompt_group_reward_credits = group_reward_credits[start:end]
                prompt_adv, prompt_group_adv, prompt_rollout_adv, prompt_metrics = compute_sgrpo_advantages(
                    rollout_rewards=prompt_rollout_rewards,
                    group_rewards=prompt_group_rewards,
                    num_generations=self.config.num_generations,
                    supergroup_num_groups=self.config.supergroup_num_groups,
                    group_advantage_weight=self.config.group_advantage_weight,
                    scale_rewards=False,
                    hierarchy=self.config.hierarchy,
                    group_mean_individual_rewards=prompt_group_mean_individual_rewards,
                    individual_reward_thresholds=self.config.individual_reward_thresholds,
                    group_rewrad_credit=self.config.group_rewrad_credit,
                    group_rewrad_credit_temperature=self.config.group_rewrad_credit_temperature,
                    group_reward_credits=prompt_group_reward_credits,
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

    def _reset_cuda_phase_peak(self):
        if self.device.type != 'cuda':
            return
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)

    def _capture_cuda_phase_peak(self, phase_name):
        if self.device.type != 'cuda':
            raise RuntimeError('CUDA phase peak capture requires a CUDA device')
        if phase_name not in self._cuda_phase_run_max_reserved:
            raise ValueError(f'Unsupported CUDA phase name: {phase_name}')
        torch.cuda.synchronize(self.device)
        total_memory = float(torch.cuda.get_device_properties(self.device).total_memory)
        step_peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
        step_peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
        self._cuda_phase_run_max_reserved[phase_name] = max(
            self._cuda_phase_run_max_reserved[phase_name],
            step_peak_reserved,
        )
        self._cuda_phase_run_max_allocated[phase_name] = max(
            self._cuda_phase_run_max_allocated[phase_name],
            step_peak_allocated,
        )
        metrics = {
            f'cuda_{phase_name}_step_max_reserved_gib': float(step_peak_reserved / (1024 ** 3)),
            f'cuda_{phase_name}_step_max_reserved_ratio': float(step_peak_reserved / total_memory),
            f'cuda_{phase_name}_step_max_allocated_gib': float(step_peak_allocated / (1024 ** 3)),
            f'cuda_{phase_name}_step_max_allocated_ratio': float(step_peak_allocated / total_memory),
            f'cuda_{phase_name}_run_max_reserved_gib': float(
                self._cuda_phase_run_max_reserved[phase_name] / (1024 ** 3)
            ),
            f'cuda_{phase_name}_run_max_reserved_ratio': float(
                self._cuda_phase_run_max_reserved[phase_name] / total_memory
            ),
            f'cuda_{phase_name}_run_max_allocated_gib': float(
                self._cuda_phase_run_max_allocated[phase_name] / (1024 ** 3)
            ),
            f'cuda_{phase_name}_run_max_allocated_ratio': float(
                self._cuda_phase_run_max_allocated[phase_name] / total_memory
            ),
        }
        return metrics, step_peak_reserved, step_peak_allocated

    def train(self):
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info('Starting ProGen2 SGRPO training in %s', self.output_dir)
        self.calibrate()

        for step_idx in range(self.config.max_steps):
            logger.info('Starting train step %d/%d', step_idx + 1, self.config.max_steps)
            rollout_phase_metrics = {}
            reward_phase_metrics = {}
            training_phase_metrics = {}
            rollout_step_peak_reserved = 0
            rollout_step_peak_allocated = 0
            reward_step_peak_reserved = 0
            reward_step_peak_allocated = 0
            training_step_peak_reserved = 0
            training_step_peak_allocated = 0

            if self.device.type == 'cuda':
                self._reset_cuda_phase_peak()
            prompts = self._next_prompt_batch()
            rollout = self._generate_rollouts(
                prompts,
                num_return_sequences=self.num_return_sequences,
                seed=self.config.seed + step_idx,
            )
            if self.device.type == 'cuda':
                rollout_phase_metrics, rollout_step_peak_reserved, rollout_step_peak_allocated = (
                    self._capture_cuda_phase_peak('rollout')
                )

            if self.device.type == 'cuda':
                self._reset_cuda_phase_peak()
            rollout_rewards, reward_metrics, individual_reward_tensors = self._score_rollout_rewards(
                rollout.protein_sequences,
                step_number=self.global_step + 1,
            )
            group_rewards = self._score_group_rewards(rollout.protein_sequences)
            group_reward_credits = None
            if self.config.rl_algorithm == 'sgrpo' and self.config.group_rewrad_credit == 'loo':
                group_reward_credits = self._score_group_reward_credits(rollout.protein_sequences)
            group_mean_individual_rewards = None
            if any(threshold is not None for threshold in self.config.individual_reward_thresholds.values()):
                group_mean_individual_rewards = self._build_group_mean_individual_rewards(individual_reward_tensors)
            advantages, group_advantages, rollout_advantages, advantage_metrics = self._compute_advantages(
                rollout_rewards,
                group_rewards,
                group_mean_individual_rewards=group_mean_individual_rewards,
                group_reward_credits=group_reward_credits,
            )
            if self.device.type == 'cuda':
                reward_phase_metrics, reward_step_peak_reserved, reward_step_peak_allocated = (
                    self._capture_cuda_phase_peak('reward')
                )

            if self.device.type == 'cuda':
                self._reset_cuda_phase_peak()
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
            if self.device.type == 'cuda':
                training_phase_metrics, training_step_peak_reserved, training_step_peak_allocated = (
                    self._capture_cuda_phase_peak('training')
                )

            self.global_step += 1
            metrics = {
                'loss': float(loss.detach().item()),
                'reward_mean': float(rollout_rewards.mean().item()),
                'group_reward_mean': float(advantage_metrics.get('group_reward_mean', group_rewards.mean().item())),
                'group_reward_raw_mean': float(advantage_metrics.get('group_reward_raw_mean', group_rewards.mean().item())),
                'group_reward_indicator_mean': float(advantage_metrics.get('group_reward_indicator_mean', 1.0)),
                'advantage_mean': float(advantages.mean().item()),
                'group_advantage_mean': float(group_advantages.mean().item()),
                'rollout_advantage_mean': float(rollout_advantages.mean().item()),
                **{key: float(value) for key, value in reward_metrics.items()},
                **{key: float(value) for key, value in advantage_metrics.items()},
                **{key: float(value.item() if hasattr(value, 'item') else value) for key, value in loss_metrics.items()},
                **rollout_phase_metrics,
                **reward_phase_metrics,
                **training_phase_metrics,
            }
            if self.device.type == 'cuda':
                total_memory = float(torch.cuda.get_device_properties(self.device).total_memory)
                step_peak_reserved = max(
                    rollout_step_peak_reserved,
                    reward_step_peak_reserved,
                    training_step_peak_reserved,
                )
                step_peak_allocated = max(
                    rollout_step_peak_allocated,
                    reward_step_peak_allocated,
                    training_step_peak_allocated,
                )
                self._cuda_run_max_reserved = max(self._cuda_run_max_reserved, step_peak_reserved)
                self._cuda_run_max_allocated = max(self._cuda_run_max_allocated, step_peak_allocated)
                metrics.update(
                    {
                        'cuda_step_max_reserved_gib': float(step_peak_reserved / (1024 ** 3)),
                        'cuda_step_max_reserved_ratio': float(step_peak_reserved / total_memory),
                        'cuda_step_max_allocated_gib': float(step_peak_allocated / (1024 ** 3)),
                        'cuda_step_max_allocated_ratio': float(step_peak_allocated / total_memory),
                        'cuda_run_max_reserved_gib': float(self._cuda_run_max_reserved / (1024 ** 3)),
                        'cuda_run_max_reserved_ratio': float(self._cuda_run_max_reserved / total_memory),
                        'cuda_run_max_allocated_gib': float(self._cuda_run_max_allocated / (1024 ** 3)),
                        'cuda_run_max_allocated_ratio': float(self._cuda_run_max_allocated / total_memory),
                    }
                )
            if self.global_step % self.config.logging_steps == 0:
                self._log(metrics)
            if self.global_step % self.config.save_steps == 0:
                self.save_checkpoint()

        if self.config.report_to:
            self.accelerator.end_training()
