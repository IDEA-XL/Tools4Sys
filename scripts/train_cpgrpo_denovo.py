# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime

sys.path.append(os.path.realpath('.'))

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from genmol.rl.cpgrpo import compute_clipped_grpo_loss, compute_leave_one_out_advantages
from genmol.rl.policy import GenMolCpGRPOPolicy
from genmol.rl.reward import MolecularReward
from genmol.rl.specs import sample_group_specs


@dataclass
class TrainConfig:
    init_ckpt_path: str
    ref_ckpt_path: str | None = None
    output_dir: str | None = None
    seed: int = 42
    num_generations: int = 8
    num_iterations: int = 2
    groups_per_rank: int = 2
    softmax_temp: float = 1.0
    randomness: float = 0.3
    min_add_len: int = 60
    beta: float = 0.01
    clip_range: float = 0.5
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    max_grad_norm: float = 0.2
    max_updates: int = 1
    save_every_updates: int = 1
    precision: str = 'bf16'
    scale_rewards: bool = False
    log_every_updates: int = 1


def load_config(path):
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    config = TrainConfig(**raw)
    if config.ref_ckpt_path is None:
        config.ref_ckpt_path = config.init_ckpt_path
    if config.num_generations <= 1:
        raise ValueError('num_generations must be greater than 1')
    if config.num_iterations <= 0:
        raise ValueError('num_iterations must be positive')
    if config.groups_per_rank <= 0:
        raise ValueError('groups_per_rank must be positive')
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


def reduce_mean(value, device, is_distributed):
    tensor = torch.tensor(float(value), device=device)
    if is_distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor.item()


def ensure_exists(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f'{label} not found: {path}')


def write_jsonl(path, payload):
    with open(path, 'a') as handle:
        handle.write(json.dumps(payload, sort_keys=True) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    is_distributed, rank, local_rank, world_size = init_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    ensure_exists(config.init_ckpt_path, 'init checkpoint')
    ensure_exists(config.ref_ckpt_path, 'reference checkpoint')

    output_dir = resolve_output_dir(config, args.config)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    metrics_path = os.path.join(output_dir, 'metrics.jsonl')
    state_path = os.path.join(output_dir, 'train_state.json')

    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as handle:
            yaml.safe_dump(asdict(config), handle, sort_keys=False)

    if is_distributed:
        dist.barrier()

    policy = GenMolCpGRPOPolicy(
        checkpoint_path=config.init_ckpt_path,
        device=device,
        precision=config.precision,
        trainable=True,
    )
    reference = GenMolCpGRPOPolicy(
        checkpoint_path=config.ref_ckpt_path,
        device=device,
        precision=config.precision,
        trainable=False,
    )

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
    reward_model = MolecularReward()

    for update_idx in range(1, config.max_updates + 1):
        optimizer.zero_grad(set_to_none=True)
        update_seed = config.seed + update_idx * 10000 + rank * 1000
        specs = sample_group_specs(
            groups_per_rank=config.groups_per_rank,
            softmax_temp=config.softmax_temp,
            randomness=config.randomness,
            min_add_len=config.min_add_len,
            seed=update_seed,
        )

        metric_buckets = {
            'reward_mean': [],
            'reward_std': [],
            'advantage_mean': [],
            'ratio_mean': [],
            'clip_ratio': [],
            'valid_fraction': [],
            'alert_hit_fraction': [],
            'invalid_fraction': [],
        }
        if config.beta > 0.0:
            metric_buckets['kl_mean'] = []

        for group_idx, spec in enumerate(specs):
            rollout_seed = update_seed + group_idx * 100
            rollout = policy.rollout_group(
                spec=spec,
                num_samples=config.num_generations,
                seed=rollout_seed,
            )
            reward_records = reward_model.score(rollout.smiles)
            rewards = torch.tensor([record.reward for record in reward_records], device=device, dtype=torch.float32)
            advantages = compute_leave_one_out_advantages(
                rewards=rewards,
                group_size=config.num_generations,
                scale_rewards=config.scale_rewards,
            )

            score_seed = rollout_seed + 17
            old_log_probs, _ = policy.coupled_log_probs(
                token_ids=rollout.token_ids,
                completion_mask=rollout.completion_mask,
                num_iterations=config.num_iterations,
                base_seed=score_seed,
                requires_grad=False,
            )
            ref_log_probs = None
            if config.beta > 0.0:
                ref_log_probs, _ = reference.coupled_log_probs(
                    token_ids=rollout.token_ids,
                    completion_mask=rollout.completion_mask,
                    num_iterations=config.num_iterations,
                    base_seed=score_seed,
                    requires_grad=False,
                )

            new_log_probs, _ = policy.coupled_log_probs(
                token_ids=rollout.token_ids,
                completion_mask=rollout.completion_mask,
                num_iterations=config.num_iterations,
                base_seed=score_seed,
                requires_grad=True,
            )

            loss, cpgrpo_metrics = compute_clipped_grpo_loss(
                new_log_probs=new_log_probs,
                old_log_probs=old_log_probs,
                advantages=advantages,
                completion_mask=rollout.completion_mask,
                clip_range=config.clip_range,
                ref_log_probs=ref_log_probs,
                beta=config.beta,
            )
            (loss / config.groups_per_rank).backward()

            valid_fraction = sum(record.is_valid for record in reward_records) / len(reward_records)
            alert_hit_fraction = sum(record.alert_hit for record in reward_records) / len(reward_records)
            invalid_fraction = sum(not record.is_valid for record in reward_records) / len(reward_records)

            metric_buckets['reward_mean'].append(rewards.mean().item())
            metric_buckets['reward_std'].append(rewards.std(unbiased=False).item())
            metric_buckets['advantage_mean'].append(advantages.mean().item())
            metric_buckets['ratio_mean'].append(cpgrpo_metrics['ratio_mean'])
            metric_buckets['clip_ratio'].append(cpgrpo_metrics['clip_ratio'])
            metric_buckets['valid_fraction'].append(valid_fraction)
            metric_buckets['alert_hit_fraction'].append(alert_hit_fraction)
            metric_buckets['invalid_fraction'].append(invalid_fraction)
            if 'kl_mean' in cpgrpo_metrics:
                metric_buckets['kl_mean'].append(cpgrpo_metrics['kl_mean'])

        grad_norm = torch.nn.utils.clip_grad_norm_(list(policy.trainable_parameters()), config.max_grad_norm)
        optimizer.step()
        policy.update_ema()

        reduced_metrics = {}
        for key, values in metric_buckets.items():
            reduced_metrics[key] = reduce_mean(sum(values) / len(values), device=device, is_distributed=is_distributed)
        reduced_metrics['grad_norm'] = reduce_mean(float(grad_norm), device=device, is_distributed=is_distributed)
        reduced_metrics['update'] = update_idx
        reduced_metrics['world_size'] = world_size

        if rank == 0 and update_idx % config.log_every_updates == 0:
            log_line = ' '.join(
                [
                    f'update={update_idx}',
                    f'reward_mean={reduced_metrics["reward_mean"]:.6f}',
                    f'reward_std={reduced_metrics["reward_std"]:.6f}',
                    f'advantage_mean={reduced_metrics["advantage_mean"]:.6f}',
                    f'ratio_mean={reduced_metrics["ratio_mean"]:.6f}',
                    f'clip_ratio={reduced_metrics["clip_ratio"]:.6f}',
                    f'valid_fraction={reduced_metrics["valid_fraction"]:.6f}',
                    f'alert_hit_fraction={reduced_metrics["alert_hit_fraction"]:.6f}',
                    f'invalid_fraction={reduced_metrics["invalid_fraction"]:.6f}',
                    f'grad_norm={reduced_metrics["grad_norm"]:.6f}',
                ]
            )
            print(log_line, flush=True)
            write_jsonl(metrics_path, reduced_metrics)
            with open(state_path, 'w') as handle:
                json.dump(reduced_metrics, handle, sort_keys=True, indent=2)

        if rank == 0 and update_idx % config.save_every_updates == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'update_{update_idx:06d}.ckpt')
            policy.save_checkpoint(checkpoint_path, step=update_idx)

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
