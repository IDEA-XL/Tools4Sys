# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.realpath('.'), 'src'))

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from genmol.rl.cpgrpo import (
    compute_clipped_grpo_loss,
    compute_grouped_advantages,
    compute_warmup_steps,
    split_tensor_dict,
)
from genmol.rl.policy import GenMolCpGRPOPolicy
from genmol.rl.reward import MolecularReward
from genmol.rl.specs import deserialize_specs, expand_group_specs, sample_group_specs, serialize_specs


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


def load_config(path):
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    config = TrainConfig(**raw)
    if config.ref_ckpt_path is None:
        config.ref_ckpt_path = config.init_ckpt_path
    if config.do_eval:
        raise ValueError('do_eval=True is not supported for GenMol cpGRPO')
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


def init_distributed():
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        return False, 0, 0, 1

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    if not torch.cuda.is_available():
        raise RuntimeError('Distributed training requires CUDA')

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    return True, rank, local_rank, world_size


def resolve_output_dir(config, config_path):
    if config.output_dir is not None:
        return config.output_dir

    cluster_root = '/public/home/xinwuye/ai4s-tool-joint-train'
    if os.path.isdir(cluster_root):
        base_dir = os.path.join(cluster_root, 'runs', 'cpgrpo_denovo')
    else:
        repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        base_dir = os.path.join(repo_root, 'runs', 'cpgrpo_denovo')

    config_name = os.path.splitext(os.path.basename(config_path))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(base_dir, f'{config_name}_{timestamp}')


def ensure_exists(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f'{label} not found: {path}')


def ensure_output_dir(path, overwrite):
    if os.path.exists(path):
        if not overwrite and os.listdir(path):
            raise FileExistsError(f'output_dir already exists and is non-empty: {path}')
    else:
        os.makedirs(path, exist_ok=True)


def reduce_mean(value, device, is_distributed):
    tensor = torch.tensor(float(value), device=device)
    if is_distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor.item()


def all_gather_tensor(tensor, is_distributed):
    if not is_distributed:
        return tensor
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def broadcast_specs(group_specs, num_generations, rank, is_distributed):
    payload = [None]
    if rank == 0:
        payload[0] = serialize_specs(expand_group_specs(group_specs, num_generations))
    if is_distributed:
        dist.broadcast_object_list(payload, src=0)
    return deserialize_specs(payload[0])


def write_jsonl(path, payload):
    with open(path, 'a') as handle:
        handle.write(json.dumps(payload, sort_keys=True) + '\n')


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


def maybe_trim_checkpoints(checkpoint_dir, save_total_limit):
    if save_total_limit is None or save_total_limit <= 0:
        return
    checkpoints = sorted(
        [
            os.path.join(checkpoint_dir, name)
            for name in os.listdir(checkpoint_dir)
            if name.endswith('.ckpt')
        ]
    )
    while len(checkpoints) > save_total_limit:
        os.remove(checkpoints.pop(0))


def sample_mask_seeds(config, device):
    if config.random_masking:
        return torch.randint(0, 2**12, (config.num_iterations,), device=device).tolist()
    return [42] * config.num_iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    is_distributed, rank, local_rank, world_size = init_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    ensure_exists(config.init_ckpt_path, 'init checkpoint')
    ensure_exists(config.ref_ckpt_path, 'reference checkpoint')

    local_sample_count = config.per_device_train_batch_size * config.gradient_accumulation_steps
    global_sample_count = local_sample_count * world_size
    if global_sample_count % config.num_generations != 0:
        raise ValueError(
            'global train batch size must be divisible by num_generations: '
            f'{global_sample_count} vs {config.num_generations}'
        )
    num_groups_global = global_sample_count // config.num_generations

    output_dir = resolve_output_dir(config, args.config)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    metrics_path = os.path.join(output_dir, 'metrics.jsonl')
    state_path = os.path.join(output_dir, 'train_state.json')

    if rank == 0:
        ensure_output_dir(output_dir, config.overwrite_output_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as handle:
            yaml.safe_dump(asdict(config), handle, sort_keys=False)
        print(f'output_dir={output_dir}', flush=True)
        print(f'init_ckpt_path={config.init_ckpt_path}', flush=True)
        print(f'ref_ckpt_path={config.ref_ckpt_path}', flush=True)
        print(f'world_size={world_size}', flush=True)
        print(f'local_sample_count={local_sample_count}', flush=True)
        print(f'global_sample_count={global_sample_count}', flush=True)

    if is_distributed:
        dist.barrier(device_ids=[local_rank])

    policy = GenMolCpGRPOPolicy(
        checkpoint_path=config.init_ckpt_path,
        device=device,
        bf16=config.bf16,
        trainable=True,
    )
    reference = GenMolCpGRPOPolicy(
        checkpoint_path=config.ref_ckpt_path,
        device=device,
        bf16=config.bf16,
        trainable=False,
    )

    if config.gradient_checkpointing:
        policy.enable_gradient_checkpointing(config.gradient_checkpointing_kwargs)
    policy.train()

    if is_distributed:
        policy.model.backbone = DDP(
            policy.model.backbone,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(
        policy.trainable_parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )
    scheduler = build_scheduler(optimizer, config)
    reward_model = MolecularReward()

    optimizer_step_idx = 0
    generation_cycle_idx = 0

    while optimizer_step_idx < config.max_steps:
        cycle_seed = config.seed + generation_cycle_idx * 10000
        if rank == 0:
            group_specs = sample_group_specs(
                num_groups=num_groups_global,
                generation_temperature=config.generation_temperature,
                randomness=config.randomness,
                min_add_len=config.min_add_len,
                max_completion_length=config.max_completion_length,
                seed=cycle_seed,
            )
        else:
            group_specs = None
        expanded_specs = broadcast_specs(
            group_specs=group_specs if group_specs is not None else [],
            num_generations=config.num_generations,
            rank=rank,
            is_distributed=is_distributed,
        )

        local_start = rank * local_sample_count
        local_end = (rank + 1) * local_sample_count
        local_specs = expanded_specs[local_start:local_end]
        rollout_seed = cycle_seed + rank * 1000
        rollout = policy.rollout_specs(
            specs=local_specs,
            generation_batch_size=config.generation_batch_size,
            seed=rollout_seed,
        )

        reward_records = reward_model.score(rollout.smiles)
        local_rewards = torch.tensor([record.reward for record in reward_records], device=device, dtype=torch.float32)
        global_rewards = all_gather_tensor(local_rewards, is_distributed=is_distributed)
        global_advantages, global_reward_std, zero_std_ratio = compute_grouped_advantages(
            rewards=global_rewards,
            num_generations=config.num_generations,
            scale_rewards=config.scale_rewards,
        )
        local_advantages = global_advantages[local_start:local_end].to(device=device)

        mask_seeds = sample_mask_seeds(config, device=device)
        if config.num_iterations > 1:
            expanded_input_ids = rollout.token_ids.unsqueeze(0).expand(config.num_iterations, -1, -1)
            old_per_token_logps = policy.per_token_logps(
                input_ids=expanded_input_ids,
                completion_mask=rollout.completion_mask,
                mask_seeds=mask_seeds,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                requires_grad=False,
            )
        else:
            old_per_token_logps = None

        if config.beta == 0.0:
            ref_per_token_logps = None
        else:
            expanded_input_ids = rollout.token_ids.unsqueeze(0).expand(config.num_iterations, -1, -1)
            ref_per_token_logps = reference.per_token_logps(
                input_ids=expanded_input_ids,
                completion_mask=rollout.completion_mask,
                mask_seeds=mask_seeds,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                requires_grad=False,
            )

        buffered_inputs = {
            'token_ids': rollout.token_ids,
            'completion_mask': rollout.completion_mask,
            'advantages': local_advantages,
            'old_per_token_logps': old_per_token_logps,
            'ref_per_token_logps': ref_per_token_logps,
            'mask_seeds': mask_seeds,
        }
        micro_batches = split_tensor_dict(buffered_inputs, config.gradient_accumulation_steps)

        cycle_reward_mean = global_rewards.mean().item()
        cycle_reward_std = global_reward_std.mean().item()
        cycle_valid_fraction = sum(record.is_valid for record in reward_records) / len(reward_records)
        cycle_alert_hit_fraction = sum(record.alert_hit for record in reward_records) / len(reward_records)
        cycle_invalid_fraction = sum(not record.is_valid for record in reward_records) / len(reward_records)
        completion_length = rollout.completion_mask.sum(dim=1).float().mean().item()

        for iteration_idx in range(config.num_iterations):
            if optimizer_step_idx >= config.max_steps:
                break

            optimizer.zero_grad(set_to_none=True)
            metric_buckets = defaultdict(list)

            for accum_idx, inputs in enumerate(micro_batches):
                current_seed = [inputs['mask_seeds'][iteration_idx]]
                new_per_token_logps = policy.per_token_logps(
                    input_ids=inputs['token_ids'].unsqueeze(0),
                    completion_mask=inputs['completion_mask'],
                    mask_seeds=current_seed,
                    gradient_accumulation_steps=config.gradient_accumulation_steps,
                    requires_grad=True,
                )

                if inputs['old_per_token_logps'] is None:
                    old_slice = new_per_token_logps.detach()
                else:
                    old_slice = inputs['old_per_token_logps'][:, iteration_idx:iteration_idx + 1, :]

                if inputs['ref_per_token_logps'] is None:
                    ref_slice = None
                else:
                    ref_slice = inputs['ref_per_token_logps'][:, iteration_idx:iteration_idx + 1, :]

                loss, step_metrics = compute_clipped_grpo_loss(
                    new_log_probs=new_per_token_logps,
                    old_log_probs=old_slice,
                    advantages=inputs['advantages'],
                    completion_mask=inputs['completion_mask'],
                    epsilon=config.epsilon,
                    ref_log_probs=ref_slice,
                    beta=config.beta,
                )
                (loss / config.gradient_accumulation_steps).backward()

                metric_buckets['ratio_mean'].append(step_metrics['ratio_mean'])
                metric_buckets['clip_ratio_low_mean'].append(step_metrics['clip_ratio_low_mean'])
                metric_buckets['clip_ratio_high_mean'].append(step_metrics['clip_ratio_high_mean'])
                metric_buckets['clip_ratio_region_mean'].append(step_metrics['clip_ratio_region_mean'])
                if 'kl_mean' in step_metrics:
                    metric_buckets['kl_mean'].append(step_metrics['kl_mean'])

            grad_norm = torch.nn.utils.clip_grad_norm_(list(policy.trainable_parameters()), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            policy.update_ema()
            optimizer_step_idx += 1

            if config.sync_ref_model and optimizer_step_idx % config.ref_model_sync_steps == 0:
                reference.sync_from(policy, alpha=config.ref_model_mixup_alpha)

            reduced_metrics = {
                'step': optimizer_step_idx,
                'cycle': generation_cycle_idx,
                'iteration_idx': iteration_idx,
                'reward_mean': reduce_mean(cycle_reward_mean, device, is_distributed),
                'reward_std': reduce_mean(cycle_reward_std, device, is_distributed),
                'advantage_mean': reduce_mean(local_advantages.mean().item(), device, is_distributed),
                'zero_std_ratio': reduce_mean(zero_std_ratio, device, is_distributed),
                'completion_length': reduce_mean(completion_length, device, is_distributed),
                'valid_fraction': reduce_mean(cycle_valid_fraction, device, is_distributed),
                'alert_hit_fraction': reduce_mean(cycle_alert_hit_fraction, device, is_distributed),
                'invalid_fraction': reduce_mean(cycle_invalid_fraction, device, is_distributed),
                'ratio_mean': reduce_mean(sum(metric_buckets['ratio_mean']) / len(metric_buckets['ratio_mean']), device, is_distributed),
                'clip_ratio_low_mean': reduce_mean(sum(metric_buckets['clip_ratio_low_mean']) / len(metric_buckets['clip_ratio_low_mean']), device, is_distributed),
                'clip_ratio_high_mean': reduce_mean(sum(metric_buckets['clip_ratio_high_mean']) / len(metric_buckets['clip_ratio_high_mean']), device, is_distributed),
                'clip_ratio_region_mean': reduce_mean(sum(metric_buckets['clip_ratio_region_mean']) / len(metric_buckets['clip_ratio_region_mean']), device, is_distributed),
                'grad_norm': reduce_mean(float(grad_norm), device, is_distributed),
                'lr': scheduler.get_last_lr()[0],
            }
            if 'kl_mean' in metric_buckets:
                reduced_metrics['kl_mean'] = reduce_mean(sum(metric_buckets['kl_mean']) / len(metric_buckets['kl_mean']), device, is_distributed)

            should_log = optimizer_step_idx == 1 and config.logging_first_step
            should_log = should_log or (optimizer_step_idx % config.logging_steps == 0)
            if rank == 0 and should_log:
                log_parts = [
                    f"step={optimizer_step_idx}",
                    f"reward_mean={reduced_metrics['reward_mean']:.6f}",
                    f"reward_std={reduced_metrics['reward_std']:.6f}",
                    f"advantage_mean={reduced_metrics['advantage_mean']:.6f}",
                    f"ratio_mean={reduced_metrics['ratio_mean']:.6f}",
                    f"clip_ratio_low_mean={reduced_metrics['clip_ratio_low_mean']:.6f}",
                    f"clip_ratio_high_mean={reduced_metrics['clip_ratio_high_mean']:.6f}",
                    f"clip_ratio_region_mean={reduced_metrics['clip_ratio_region_mean']:.6f}",
                    f"completion_length={reduced_metrics['completion_length']:.6f}",
                    f"zero_std_ratio={reduced_metrics['zero_std_ratio']:.6f}",
                    f"valid_fraction={reduced_metrics['valid_fraction']:.6f}",
                    f"alert_hit_fraction={reduced_metrics['alert_hit_fraction']:.6f}",
                    f"invalid_fraction={reduced_metrics['invalid_fraction']:.6f}",
                    f"grad_norm={reduced_metrics['grad_norm']:.6f}",
                    f"lr={reduced_metrics['lr']:.8f}",
                ]
                if 'kl_mean' in reduced_metrics:
                    log_parts.append(f"kl_mean={reduced_metrics['kl_mean']:.6f}")
                print(' '.join(log_parts), flush=True)
                write_jsonl(metrics_path, reduced_metrics)
                with open(state_path, 'w') as handle:
                    json.dump(reduced_metrics, handle, sort_keys=True, indent=2)

            if rank == 0 and optimizer_step_idx % config.save_steps == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'step_{optimizer_step_idx:06d}.ckpt')
                policy.save_checkpoint(checkpoint_path, step=optimizer_step_idx)
                maybe_trim_checkpoints(checkpoint_dir, config.save_total_limit)

        generation_cycle_idx += 1

    if is_distributed:
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
