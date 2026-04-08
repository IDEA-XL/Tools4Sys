import json
import logging
import math
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime

import torch
import torch.distributed as dist
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed

from genmol.rl.cpgrpo import (
    compute_clipped_grpo_loss,
    compute_grouped_advantages,
    compute_warmup_steps,
    split_tensor_dict,
)
from genmol.rl.policy import GenMolCpGRPOPolicy
from genmol.rl.reward import MolecularReward
from genmol.rl.specs import deserialize_specs, expand_group_specs, sample_group_specs, serialize_specs


logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    init_ckpt_path: str
    ref_ckpt_path: str | None = None
    output_dir: str | None = None
    overwrite_output_dir: bool = False
    seed: int = 42
    sync_ref_model: bool = True
    ref_model_sync_steps: int = 64
    ref_model_mixup_alpha: float = 0.6
    beta: float = 0.01
    epsilon: float = 0.5
    scale_rewards: bool = False
    bf16: bool = True
    do_eval: bool = False
    num_generations: int = 8
    num_iterations: int = 2
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict = field(default_factory=lambda: {'use_reentrant': False})
    learning_rate: float = 1e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    adam_eps: float = 1e-8
    weight_decay: float = 0.1
    max_grad_norm: float = 0.2
    logging_first_step: bool = True
    logging_steps: int = 10
    logging_strategy: str = 'steps'
    max_steps: int = 100
    lr_scheduler_type: str = 'cosine_with_min_lr'
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {'min_lr_rate': 0.1})
    warmup_ratio: float = 0.0001
    save_strategy: str = 'steps'
    save_steps: int = 50
    save_total_limit: int = 5
    random_masking: bool = True
    generation_batch_size: int = 16
    generation_temperature: float = 1.0
    randomness: float = 0.3
    min_add_len: int = 60
    max_completion_length: int | None = None
    log_completions: bool = True
    log_level: str = 'info'
    report_to: list[str] = field(default_factory=list)


@dataclass
class TrainResult:
    metrics: dict


def load_config(path):
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    config = TrainConfig(**raw)
    if config.ref_ckpt_path is None:
        config.ref_ckpt_path = config.init_ckpt_path
    if config.logging_strategy != 'steps':
        raise ValueError('Only logging_strategy=steps is supported')
    if config.save_strategy != 'steps':
        raise ValueError('Only save_strategy=steps is supported')
    if config.num_generations <= 1:
        raise ValueError('num_generations must be greater than 1')
    if config.num_iterations <= 0:
        raise ValueError('num_iterations must be positive')
    if config.per_device_train_batch_size <= 0:
        raise ValueError('per_device_train_batch_size must be positive')
    if config.gradient_accumulation_steps <= 0:
        raise ValueError('gradient_accumulation_steps must be positive')
    if config.generation_batch_size <= 0:
        raise ValueError('generation_batch_size must be positive')
    if not 0.0 <= config.ref_model_mixup_alpha <= 1.0:
        raise ValueError('ref_model_mixup_alpha must be in [0, 1]')
    return config


def resolve_output_dir(config, config_path):
    if config.output_dir is not None:
        return config.output_dir

    cluster_root = '/public/home/xinwuye/ai4s-tool-joint-train'
    if os.path.isdir(cluster_root):
        base_dir = os.path.join(cluster_root, 'runs', 'cpgrpo_denovo')
    else:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
        base_dir = os.path.join(repo_root, 'runs', 'cpgrpo_denovo')

    config_name = os.path.splitext(os.path.basename(config_path))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(base_dir, f'{config_name}_{timestamp}')


def ensure_exists(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f'{label} not found: {path}')


def build_scheduler(optimizer, config):
    if config.lr_scheduler_type != 'cosine_with_min_lr':
        raise ValueError(f'Unsupported lr_scheduler_type: {config.lr_scheduler_type}')

    min_lr_rate = float(config.lr_scheduler_kwargs.get('min_lr_rate', 0.1))
    warmup_steps = compute_warmup_steps(config.max_steps, config.warmup_ratio)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, config.max_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_rate + (1.0 - min_lr_rate) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def broadcast_specs(group_specs, num_generations, accelerator):
    payload = [None]
    if accelerator.is_main_process:
        payload[0] = serialize_specs(expand_group_specs(group_specs, num_generations))
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    return deserialize_specs(payload[0])


def write_jsonl(path, payload):
    with open(path, 'a') as handle:
        handle.write(json.dumps(payload, sort_keys=True) + '\n')


def _nanmean(tensor):
    if tensor.numel() == 0 or torch.isnan(tensor).all():
        return float('nan')
    return torch.nanmean(tensor).item()


def _nanreduce(tensor, mode):
    valid = tensor[~torch.isnan(tensor)]
    if valid.numel() == 0:
        return float('nan')
    if mode == 'min':
        return valid.min().item()
    if mode == 'max':
        return valid.max().item()
    raise ValueError(f'Unsupported nan reduction mode: {mode}')


def find_last_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None

    pattern = re.compile(r'^checkpoint-(\d+)$')
    matches = []
    for name in os.listdir(output_dir):
        match = pattern.match(name)
        if match is not None and os.path.isdir(os.path.join(output_dir, name)):
            matches.append((int(match.group(1)), os.path.join(output_dir, name)))
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


def maybe_trim_checkpoints(output_dir, save_total_limit):
    if save_total_limit is None or save_total_limit <= 0:
        return

    pattern = re.compile(r'^checkpoint-(\d+)$')
    checkpoints = []
    for name in os.listdir(output_dir):
        match = pattern.match(name)
        if match is not None and os.path.isdir(os.path.join(output_dir, name)):
            checkpoints.append((int(match.group(1)), os.path.join(output_dir, name)))
    checkpoints.sort()
    while len(checkpoints) > save_total_limit:
        _, path = checkpoints.pop(0)
        for root, dirs, files in os.walk(path, topdown=False):
            for filename in files:
                os.remove(os.path.join(root, filename))
            for dirname in dirs:
                os.rmdir(os.path.join(root, dirname))
        os.rmdir(path)


class GenMolCpGRPOTrainer:
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        self.accelerator = Accelerator(
            log_with=config.report_to if config.report_to else None,
            mixed_precision='bf16' if config.bf16 else 'no',
        )
        self.device = self.accelerator.device
        self.world_size = self.accelerator.num_processes
        self.local_sample_count = config.per_device_train_batch_size * config.gradient_accumulation_steps
        self.global_sample_count = self.local_sample_count * self.world_size
        if self.global_sample_count % config.num_generations != 0:
            raise ValueError(
                'global train batch size must be divisible by num_generations: '
                f'{self.global_sample_count} vs {config.num_generations}'
            )
        self.num_groups_global = self.global_sample_count // config.num_generations

        deepspeed_plugin = getattr(self.accelerator.state, 'deepspeed_plugin', None)
        if deepspeed_plugin is not None:
            deepspeed_config = deepspeed_plugin.deepspeed_config
            deepspeed_config['train_micro_batch_size_per_gpu'] = int(config.per_device_train_batch_size)
            deepspeed_config['gradient_accumulation_steps'] = int(config.gradient_accumulation_steps)
            deepspeed_config['train_batch_size'] = int(self.global_sample_count)

        ensure_exists(config.init_ckpt_path, 'init checkpoint')
        ensure_exists(config.ref_ckpt_path, 'reference checkpoint')

        set_seed(config.seed, device_specific=True)

        self.policy = GenMolCpGRPOPolicy(
            checkpoint_path=config.init_ckpt_path,
            device=self.device,
            bf16=config.bf16,
            trainable=True,
        )
        self.reference = GenMolCpGRPOPolicy(
            checkpoint_path=config.ref_ckpt_path,
            device=self.device,
            bf16=config.bf16,
            trainable=False,
        )
        if config.gradient_checkpointing:
            self.policy.enable_gradient_checkpointing(config.gradient_checkpointing_kwargs)
        self.policy.train()

        optimizer = torch.optim.AdamW(
            self.policy.trainable_parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
        scheduler = build_scheduler(optimizer, config)
        self.policy.model.backbone, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.policy.model.backbone,
            optimizer,
            scheduler,
        )

        self.reward_model = MolecularReward()
        self.metrics_path = os.path.join(output_dir, 'metrics.jsonl')
        self.text_logs_path = os.path.join(output_dir, 'completions.jsonl')
        self.state_path = os.path.join(output_dir, 'trainer_state.json')

        self.global_step = 0
        self.generation_cycle_idx = 0
        self._step = 0
        self._buffered_inputs = None
        self._buffer_metadata = None
        self._last_train_metrics = None

        if config.report_to:
            init_kwargs = {}
            if 'wandb' in config.report_to:
                init_kwargs['wandb'] = {'name': os.path.basename(output_dir)}
            self.accelerator.init_trackers('genmol-cpgrpo', config=asdict(config), init_kwargs=init_kwargs or None)

        logger.info(
            'process_index=%s device=%s world_size=%s local_sample_count=%s global_sample_count=%s',
            self.accelerator.process_index,
            self.device,
            self.world_size,
            self.local_sample_count,
            self.global_sample_count,
        )

    def _sample_mask_seeds(self):
        if self.config.random_masking:
            return torch.randint(0, 2**12, (self.config.num_iterations,), device=self.device).tolist()
        return [42] * self.config.num_iterations

    def _all_gather_objects(self, payload):
        if not dist.is_available() or not dist.is_initialized():
            return payload
        gathered = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, payload)
        merged = []
        for shard in gathered:
            merged.extend(shard)
        return merged

    def _save_text_logs(self, rows):
        if not self.config.log_completions or not rows:
            return
        gathered = self._all_gather_objects(rows)
        if not self.accelerator.is_main_process:
            return
        for row in gathered:
            write_jsonl(self.text_logs_path, row)

    def _generate_and_score_completions(self):
        cycle_seed = self.config.seed + self.generation_cycle_idx * 10000
        if self.accelerator.is_main_process:
            group_specs = sample_group_specs(
                num_groups=self.num_groups_global,
                generation_temperature=self.config.generation_temperature,
                randomness=self.config.randomness,
                min_add_len=self.config.min_add_len,
                max_completion_length=self.config.max_completion_length,
                seed=cycle_seed,
            )
        else:
            group_specs = []

        expanded_specs = broadcast_specs(group_specs, self.config.num_generations, self.accelerator)
        local_start = self.accelerator.process_index * self.local_sample_count
        local_end = (self.accelerator.process_index + 1) * self.local_sample_count
        local_specs = expanded_specs[local_start:local_end]
        rollout_seed = cycle_seed + self.accelerator.process_index * 1000
        rollout = self.policy.rollout_specs(
            specs=local_specs,
            generation_batch_size=self.config.generation_batch_size,
            seed=rollout_seed,
        )

        reward_records = self.reward_model.score(rollout.smiles)
        local_rewards = torch.tensor([record.reward for record in reward_records], device=self.device, dtype=torch.float32)
        global_rewards = self.accelerator.gather(local_rewards).detach()
        global_advantages, global_reward_std, zero_std_ratio = compute_grouped_advantages(
            rewards=global_rewards,
            num_generations=self.config.num_generations,
            scale_rewards=self.config.scale_rewards,
        )
        local_advantages = global_advantages[local_start:local_end].to(device=self.device)

        local_valid = torch.tensor([float(record.is_valid) for record in reward_records], device=self.device)
        local_alert = torch.tensor([float(record.alert_hit) for record in reward_records], device=self.device)
        local_qed = torch.tensor(
            [float('nan') if record.qed is None else float(record.qed) for record in reward_records],
            device=self.device,
        )
        local_sa = torch.tensor(
            [float('nan') if record.sa is None else float(record.sa) for record in reward_records],
            device=self.device,
        )
        local_sa_score = torch.tensor(
            [float('nan') if record.sa_score is None else float(record.sa_score) for record in reward_records],
            device=self.device,
        )
        local_soft = torch.tensor(
            [float('nan') if record.soft_reward is None else float(record.soft_reward) for record in reward_records],
            device=self.device,
        )
        local_lengths = rollout.completion_mask.sum(dim=1).float()

        gathered_valid = self.accelerator.gather(local_valid)
        gathered_alert = self.accelerator.gather(local_alert)
        gathered_qed = self.accelerator.gather(local_qed)
        gathered_sa = self.accelerator.gather(local_sa)
        gathered_sa_score = self.accelerator.gather(local_sa_score)
        gathered_soft = self.accelerator.gather(local_soft)
        gathered_lengths = self.accelerator.gather(local_lengths)
        gathered_advantages = self.accelerator.gather(local_advantages.detach())

        mask_seeds = self._sample_mask_seeds()
        prompt_completion_ids = torch.cat([rollout.prompt_ids, rollout.completion_ids], dim=1)
        logits_to_keep = rollout.completion_ids.size(1)
        expanded_ids = prompt_completion_ids.unsqueeze(0).expand(self.config.num_iterations, -1, -1)
        if self.config.num_iterations > 1:
            old_per_token_logps = self.policy.per_token_logps(
                input_ids=expanded_ids,
                logits_to_keep=logits_to_keep,
                completion_mask=rollout.completion_mask,
                mask_seeds=mask_seeds,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                requires_grad=False,
            )
        else:
            old_per_token_logps = None

        if self.config.beta == 0.0:
            ref_per_token_logps = None
        else:
            ref_per_token_logps = self.reference.per_token_logps(
                input_ids=expanded_ids,
                logits_to_keep=logits_to_keep,
                completion_mask=rollout.completion_mask,
                mask_seeds=mask_seeds,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                requires_grad=False,
            )

        log_rows = []
        for spec, safe_string, record in zip(local_specs, rollout.safe_strings, reward_records):
            log_rows.append(
                {
                    'buffer_cycle': self.generation_cycle_idx,
                    'step': self.global_step,
                    'spec': spec.__dict__,
                    'safe': safe_string,
                    'smiles': record.smiles,
                    'reward': record.reward,
                    'qed': record.qed,
                    'sa': record.sa,
                    'sa_score': record.sa_score,
                    'soft_reward': record.soft_reward,
                    'is_valid': record.is_valid,
                    'alert_hit': record.alert_hit,
                }
            )

        metadata = {
            'buffer_cycle': self.generation_cycle_idx,
            'reward_mean': global_rewards.mean().item(),
            'reward_std': global_reward_std.mean().item(),
            'advantage_mean': gathered_advantages.mean().item(),
            'zero_std_ratio': zero_std_ratio,
            'completion_length': gathered_lengths.mean().item(),
            'valid_fraction': gathered_valid.mean().item(),
            'alert_hit_fraction': gathered_alert.mean().item(),
            'invalid_fraction': 1.0 - gathered_valid.mean().item(),
            'rewards/qed_mean': _nanmean(gathered_qed),
            'rewards/sa_mean': _nanmean(gathered_sa),
            'rewards/sa_score_mean': _nanmean(gathered_sa_score),
            'rewards/soft_mean': _nanmean(gathered_soft),
            'log_rows': log_rows,
        }

        return {
            'prompt_ids': rollout.prompt_ids,
            'completion_ids': rollout.completion_ids,
            'completion_mask': rollout.completion_mask,
            'advantages': local_advantages,
            'old_per_token_logps': old_per_token_logps,
            'ref_per_token_logps': ref_per_token_logps,
            'mask_seeds': mask_seeds,
        }, metadata

    def _prepare_inputs(self):
        generate_every = self.config.gradient_accumulation_steps * self.config.num_iterations
        if self._step % generate_every == 0 or self._buffered_inputs is None:
            accumulated_local_batch, metadata = self._generate_and_score_completions()
            self._buffered_inputs = split_tensor_dict(accumulated_local_batch, self.config.gradient_accumulation_steps)
            self._buffer_metadata = metadata
            self.generation_cycle_idx += 1

        inputs = self._buffered_inputs[self._step % self.config.gradient_accumulation_steps]
        self._step += 1
        return inputs

    def _compute_loss(self, inputs):
        this_itr_idx = self._step % self.config.num_iterations
        prompt_completion_ids = torch.cat([inputs['prompt_ids'], inputs['completion_ids']], dim=1).unsqueeze(0)
        logits_to_keep = inputs['completion_ids'].size(1)
        current_seed = [inputs['mask_seeds'][this_itr_idx]]
        per_token_logps = self.policy.per_token_logps(
            input_ids=prompt_completion_ids,
            logits_to_keep=logits_to_keep,
            completion_mask=inputs['completion_mask'],
            mask_seeds=current_seed,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            requires_grad=True,
        )
        if inputs['old_per_token_logps'] is None:
            old_per_token_logps = per_token_logps.detach()
        else:
            old_per_token_logps = inputs['old_per_token_logps'][:, this_itr_idx, :].unsqueeze(1)

        if inputs['ref_per_token_logps'] is None:
            ref_per_token_logps = None
        else:
            ref_per_token_logps = inputs['ref_per_token_logps'][:, this_itr_idx, :].unsqueeze(1)

        return compute_clipped_grpo_loss(
            new_log_probs=per_token_logps,
            old_log_probs=old_per_token_logps,
            advantages=inputs['advantages'],
            completion_mask=inputs['completion_mask'],
            epsilon=self.config.epsilon,
            ref_log_probs=ref_per_token_logps,
            beta=self.config.beta,
        )

    def _aggregate_metric(self, values, mode='mean'):
        local = torch.stack(values).mean().reshape(1).to(self.device)
        gathered = self.accelerator.gather(local)
        if mode == 'mean':
            return gathered.nanmean().item()
        if mode == 'min':
            return _nanreduce(gathered, mode='min')
        if mode == 'max':
            return _nanreduce(gathered, mode='max')
        raise ValueError(f'Unsupported aggregation mode: {mode}')

    def _build_log_metrics(self, metric_buckets, grad_norm):
        metadata = dict(self._buffer_metadata)
        metrics = {
            'step': self.global_step,
            'buffer_cycle': metadata['buffer_cycle'],
            'reward_mean': metadata['reward_mean'],
            'reward_std': metadata['reward_std'],
            'advantage_mean': metadata['advantage_mean'],
            'zero_std_ratio': metadata['zero_std_ratio'],
            'completion_length': metadata['completion_length'],
            'valid_fraction': metadata['valid_fraction'],
            'alert_hit_fraction': metadata['alert_hit_fraction'],
            'invalid_fraction': metadata['invalid_fraction'],
            'rewards/qed_mean': metadata['rewards/qed_mean'],
            'rewards/sa_mean': metadata['rewards/sa_mean'],
            'rewards/sa_score_mean': metadata['rewards/sa_score_mean'],
            'rewards/soft_mean': metadata['rewards/soft_mean'],
            'ratio_mean': self._aggregate_metric(metric_buckets['ratio_mean']),
            'clip_ratio/low_mean': self._aggregate_metric(metric_buckets['clip_ratio_low_mean']),
            'clip_ratio/low_min': self._aggregate_metric(metric_buckets['clip_ratio_low_mean'], mode='min'),
            'clip_ratio/high_mean': self._aggregate_metric(metric_buckets['clip_ratio_high_mean']),
            'clip_ratio/high_max': self._aggregate_metric(metric_buckets['clip_ratio_high_mean'], mode='max'),
            'clip_ratio/region_mean': self._aggregate_metric(metric_buckets['clip_ratio_region_mean']),
            'grad_norm': self._aggregate_metric([torch.as_tensor(float(grad_norm), device=self.device)]),
            'lr': self.scheduler.get_last_lr()[0],
        }
        if 'kl_mean' in metric_buckets:
            metrics['kl_mean'] = self._aggregate_metric(metric_buckets['kl_mean'])
        return metrics

    def _log_metrics(self, split, metrics):
        if self.accelerator.is_main_process:
            logger.info(
                '%s step=%s reward_mean=%.6f reward_std=%.6f advantage_mean=%.6f ratio_mean=%.6f '
                'clip_ratio/low_mean=%.6f clip_ratio/low_min=%.6f clip_ratio/high_mean=%.6f '
                'clip_ratio/high_max=%.6f clip_ratio/region_mean=%.6f completion_length=%.6f '
                'zero_std_ratio=%.6f valid_fraction=%.6f alert_hit_fraction=%.6f invalid_fraction=%.6f '
                'rewards/qed_mean=%.6f rewards/sa_mean=%.6f rewards/sa_score_mean=%.6f rewards/soft_mean=%.6f '
                'grad_norm=%.6f lr=%.8f%s',
                split,
                metrics['step'],
                metrics['reward_mean'],
                metrics['reward_std'],
                metrics['advantage_mean'],
                metrics['ratio_mean'],
                metrics['clip_ratio/low_mean'],
                metrics['clip_ratio/low_min'],
                metrics['clip_ratio/high_mean'],
                metrics['clip_ratio/high_max'],
                metrics['clip_ratio/region_mean'],
                metrics['completion_length'],
                metrics['zero_std_ratio'],
                metrics['valid_fraction'],
                metrics['alert_hit_fraction'],
                metrics['invalid_fraction'],
                metrics['rewards/qed_mean'],
                metrics['rewards/sa_mean'],
                metrics['rewards/sa_score_mean'],
                metrics['rewards/soft_mean'],
                metrics['grad_norm'],
                metrics['lr'],
                '' if 'kl_mean' not in metrics else f" kl_mean={metrics['kl_mean']:.6f}",
            )
            write_jsonl(self.metrics_path, metrics)
            with open(self.state_path, 'w') as handle:
                json.dump(
                    {
                        'global_step': self.global_step,
                        'micro_step': self._step,
                        'generation_cycle_idx': self.generation_cycle_idx,
                        'last_metrics': metrics,
                    },
                    handle,
                    sort_keys=True,
                    indent=2,
                )
        if self.config.report_to:
            self.accelerator.log(metrics, step=self.global_step)

    def _checkpoint_dir(self):
        return os.path.join(self.output_dir, f'checkpoint-{self.global_step:06d}')

    def _save_checkpoint(self):
        checkpoint_dir = self._checkpoint_dir()
        accelerator_state_dir = os.path.join(checkpoint_dir, 'accelerator_state')
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(accelerator_state_dir)
        self.accelerator.wait_for_everyone()

        if not self.accelerator.is_main_process:
            return

        model_path = os.path.join(checkpoint_dir, 'model.ckpt')
        ref_path = os.path.join(checkpoint_dir, 'reference_backbone.pt')
        trainer_state_path = os.path.join(checkpoint_dir, 'trainer_state.json')
        self.policy.save_checkpoint(model_path, step=self.global_step, accelerator=self.accelerator)
        torch.save(self.reference.get_backbone_state_dict(), ref_path)
        with open(trainer_state_path, 'w') as handle:
            json.dump(
                {
                    'global_step': self.global_step,
                    'micro_step': self._step,
                    'generation_cycle_idx': self.generation_cycle_idx,
                    'last_metrics': self._last_train_metrics,
                },
                handle,
                sort_keys=True,
                indent=2,
            )
        maybe_trim_checkpoints(self.output_dir, self.config.save_total_limit)

    def _load_checkpoint(self, checkpoint_dir):
        accelerator_state_dir = os.path.join(checkpoint_dir, 'accelerator_state')
        trainer_state_path = os.path.join(checkpoint_dir, 'trainer_state.json')
        model_path = os.path.join(checkpoint_dir, 'model.ckpt')
        ref_path = os.path.join(checkpoint_dir, 'reference_backbone.pt')
        ensure_exists(accelerator_state_dir, 'accelerator state')
        ensure_exists(trainer_state_path, 'trainer state')
        ensure_exists(model_path, 'model checkpoint')
        ensure_exists(ref_path, 'reference state')

        with open(trainer_state_path) as handle:
            trainer_state = json.load(handle)

        self.accelerator.load_state(accelerator_state_dir)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.policy.load_ema_state(checkpoint.get('ema'))
        self.reference.load_backbone_state_dict(torch.load(ref_path, map_location='cpu', weights_only=False))

        self.global_step = int(trainer_state['global_step'])
        self._step = int(trainer_state['micro_step'])
        self.generation_cycle_idx = int(trainer_state['generation_cycle_idx'])
        self._last_train_metrics = trainer_state.get('last_metrics')
        self._buffered_inputs = None
        self._buffer_metadata = None

    def train(self, resume_from_checkpoint=None):
        if resume_from_checkpoint is not None:
            logger.info('Resuming from checkpoint: %s', resume_from_checkpoint)
            self._load_checkpoint(resume_from_checkpoint)

        while self.global_step < self.config.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            metric_buckets = defaultdict(list)

            for _ in range(self.config.gradient_accumulation_steps):
                inputs = self._prepare_inputs()
                loss, step_metrics = self._compute_loss(inputs)
                self.accelerator.backward(loss / self.config.gradient_accumulation_steps)
                for key, value in step_metrics.items():
                    metric_buckets[key].append(value.detach())

            grad_norm = self.accelerator.clip_grad_norm_(
                self.policy.model.backbone.parameters(),
                self.config.max_grad_norm,
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.policy.update_ema()
            self.global_step += 1

            if self.config.sync_ref_model and self.global_step % self.config.ref_model_sync_steps == 0:
                self.reference.sync_from(self.policy, alpha=self.config.ref_model_mixup_alpha)

            self._last_train_metrics = self._build_log_metrics(metric_buckets, grad_norm)

            should_log = self.global_step == 1 and self.config.logging_first_step
            should_log = should_log or (self.global_step % self.config.logging_steps == 0)
            if should_log:
                self._log_metrics('train', self._last_train_metrics)
                self._save_text_logs(self._buffer_metadata['log_rows'])

            if self.global_step % self.config.save_steps == 0:
                self._save_checkpoint()

        return TrainResult(metrics=self._last_train_metrics or {})

    def evaluate(self):
        current_train_state = self.policy.model.backbone.training
        self.policy.model.backbone.eval()
        try:
            _, metadata = self._generate_and_score_completions()
        finally:
            self.policy.model.backbone.train(current_train_state)
        return {
            'step': self.global_step,
            'buffer_cycle': metadata['buffer_cycle'],
            'reward_mean': metadata['reward_mean'],
            'reward_std': metadata['reward_std'],
            'advantage_mean': metadata['advantage_mean'],
            'zero_std_ratio': metadata['zero_std_ratio'],
            'completion_length': metadata['completion_length'],
            'valid_fraction': metadata['valid_fraction'],
            'alert_hit_fraction': metadata['alert_hit_fraction'],
            'invalid_fraction': metadata['invalid_fraction'],
            'rewards/qed_mean': metadata['rewards/qed_mean'],
            'rewards/sa_mean': metadata['rewards/sa_mean'],
            'rewards/sa_score_mean': metadata['rewards/sa_score_mean'],
            'rewards/soft_mean': metadata['rewards/soft_mean'],
            'ratio_mean': float('nan'),
            'clip_ratio/low_mean': float('nan'),
            'clip_ratio/low_min': float('nan'),
            'clip_ratio/high_mean': float('nan'),
            'clip_ratio/high_max': float('nan'),
            'clip_ratio/region_mean': float('nan'),
            'grad_norm': float('nan'),
            'lr': self.scheduler.get_last_lr()[0],
        }

    def log_metrics(self, split, metrics):
        self._log_metrics(split, metrics)

    def save_metrics(self, split, metrics):
        if not self.accelerator.is_main_process:
            return
        path = os.path.join(self.output_dir, f'{split}_results.json')
        with open(path, 'w') as handle:
            json.dump(metrics, handle, sort_keys=True, indent=2)

    def save_state(self):
        if not self.accelerator.is_main_process:
            return
        with open(self.state_path, 'w') as handle:
            json.dump(
                {
                    'global_step': self.global_step,
                    'micro_step': self._step,
                    'generation_cycle_idx': self.generation_cycle_idx,
                    'last_metrics': self._last_train_metrics,
                },
                handle,
                sort_keys=True,
                indent=2,
            )

    def save_model(self, output_dir):
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            return
        self.policy.save_checkpoint(
            os.path.join(output_dir, 'final_model.ckpt'),
            step=self.global_step,
            accelerator=self.accelerator,
        )
