# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import logging
import math
import os
import random
import sys
from dataclasses import asdict, dataclass

import torch
import yaml

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.realpath('.'), 'src'))

from genmol.rl.lead_policy import LeadOptCpGRPOPolicy
from genmol.rl.lead_reward import LeadOptimizationReward
from genmol.rl.lead_specs import LeadOptSpec
from genmol.rl.policy import GenMolCpGRPOPolicy
from genmol.rl.specs import expand_group_specs, sample_group_specs
from genmol.rl.trainer import write_jsonl


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalExperimentConfig:
    name: str
    denovo_ckpt_path: str
    lead_ckpt_path: str


@dataclass(frozen=True)
class EvalConfig:
    output_markdown_path: str
    output_json_path: str | None = None
    output_rows_path: str | None = None
    seed: int = 42
    bf16: bool = True
    device: str = 'cuda'
    num_samples: int = 1024
    denovo_num_generations: int = 16
    denovo_generation_batch_size: int = 1024
    denovo_generation_temperature: float = 1.0
    denovo_randomness: float = 0.3
    denovo_min_add_len: int = 60
    denovo_max_completion_length: int | None = None
    lead_generation_batch_size: int = 256
    lead_generation_temperature: float = 1.0
    lead_randomness: float = 0.3
    lead_min_seed_len: int = 60
    sim_weight: float = 1.0
    experiments: list[EvalExperimentConfig] | None = None


def configure_logging():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )


def load_config(path):
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    experiments = [EvalExperimentConfig(**item) for item in raw.pop('experiments')]
    config = EvalConfig(experiments=experiments, **raw)
    if config.num_samples <= 0:
        raise ValueError(f'num_samples must be positive, got {config.num_samples}')
    if config.denovo_num_generations <= 0:
        raise ValueError(f'denovo_num_generations must be positive, got {config.denovo_num_generations}')
    if config.num_samples % config.denovo_num_generations != 0:
        raise ValueError(
            'num_samples must be divisible by denovo_num_generations: '
            f'{config.num_samples} vs {config.denovo_num_generations}'
        )
    if config.denovo_generation_batch_size <= 0:
        raise ValueError('denovo_generation_batch_size must be positive')
    if config.lead_generation_batch_size <= 0:
        raise ValueError('lead_generation_batch_size must be positive')
    if config.lead_min_seed_len <= 0:
        raise ValueError('lead_min_seed_len must be positive')
    if config.experiments is None or not config.experiments:
        raise ValueError('experiments must be non-empty')
    for experiment in config.experiments:
        if not experiment.name:
            raise ValueError('experiment name must be non-empty')
        if not os.path.exists(experiment.denovo_ckpt_path):
            raise FileNotFoundError(f'de novo checkpoint not found: {experiment.denovo_ckpt_path}')
        if not os.path.exists(experiment.lead_ckpt_path):
            raise FileNotFoundError(f'lead checkpoint not found: {experiment.lead_ckpt_path}')
    return config


def resolve_device(device_name):
    if device_name == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('device=cuda requested but CUDA is not available')
        return torch.device('cuda')
    return torch.device(device_name)


def _nanmean(values):
    filtered = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not filtered:
        return float('nan')
    return float(sum(filtered) / len(filtered))


def _format_metric(value):
    if value is None:
        return 'nan'
    value = float(value)
    if math.isnan(value):
        return 'nan'
    return f'{value:.6f}'


def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _invalid_lead_row(seed_smiles):
    return {
        'seed_smiles': seed_smiles,
        'smiles': None,
        'reward': -1.0,
        'base_reward': -1.0,
        'sim': None,
        'is_valid': False,
        'alert_hit': False,
        'qed': None,
        'sa': None,
        'sa_score': None,
        'soft_reward': None,
    }


def _build_markdown(results):
    columns = [
        'experiment',
        'reward_mean',
        'base_reward_mean',
        'sim_mean',
        'valid_fraction',
        'alert_hit_fraction',
        'qed_mean',
        'sa_mean',
        'sa_score_mean',
        'soft_reward_mean',
    ]
    lines = [
        '# Coupled-GRPO Pipeline Evaluation',
        '',
        '| ' + ' | '.join(columns) + ' |',
        '| ' + ' | '.join(['---'] * len(columns)) + ' |',
    ]
    for result in results:
        row = [
            result['experiment'],
            _format_metric(result['reward_mean']),
            _format_metric(result['base_reward_mean']),
            _format_metric(result['sim_mean']),
            _format_metric(result['valid_fraction']),
            _format_metric(result['alert_hit_fraction']),
            _format_metric(result['qed_mean']),
            _format_metric(result['sa_mean']),
            _format_metric(result['sa_score_mean']),
            _format_metric(result['soft_reward_mean']),
        ]
        lines.append('| ' + ' | '.join(row) + ' |')
    lines.append('')
    return '\n'.join(lines)


def evaluate_experiment(config, experiment, device):
    logger.info('Evaluating %s', experiment.name)
    denovo_policy = GenMolCpGRPOPolicy(
        checkpoint_path=experiment.denovo_ckpt_path,
        device=device,
        bf16=config.bf16,
        trainable=False,
    )
    lead_policy = LeadOptCpGRPOPolicy(
        checkpoint_path=experiment.lead_ckpt_path,
        device=device,
        bf16=config.bf16,
        trainable=False,
        score_chunk_size=max(1, config.lead_generation_batch_size),
    )
    lead_reward = LeadOptimizationReward(sim_weight=config.sim_weight)
    try:
        group_specs = sample_group_specs(
            num_groups=config.num_samples // config.denovo_num_generations,
            generation_temperature=config.denovo_generation_temperature,
            randomness=config.denovo_randomness,
            min_add_len=config.denovo_min_add_len,
            seed=config.seed,
            max_completion_length=config.denovo_max_completion_length,
        )
        denovo_specs = expand_group_specs(group_specs, config.denovo_num_generations)
        denovo_rollout = denovo_policy.rollout_specs(
            specs=denovo_specs,
            generation_batch_size=config.denovo_generation_batch_size,
            seed=config.seed,
        )

        lead_rows = [_invalid_lead_row(seed_smiles=None) for _ in range(len(denovo_rollout.smiles))]
        valid_indices = []
        valid_seed_smiles = []
        lead_specs = []
        rng = random.Random(config.seed + 7919)
        for sample_idx, seed_smiles in enumerate(denovo_rollout.smiles):
            if seed_smiles is None:
                continue
            valid_indices.append(sample_idx)
            valid_seed_smiles.append(seed_smiles)
            lead_specs.append(
                LeadOptSpec(
                    seed_smiles=seed_smiles,
                    mutation_seed=rng.randrange(2**31),
                    generation_temperature=config.lead_generation_temperature,
                    randomness=config.lead_randomness,
                    min_seed_len=config.lead_min_seed_len,
                )
            )

        if valid_indices:
            lead_rollout = lead_policy.rollout_specs(
                specs=lead_specs,
                generation_batch_size=config.lead_generation_batch_size,
                seed=config.seed + 500000,
            )
            lead_records = lead_reward.score(lead_rollout.seed_smiles, lead_rollout.smiles)
            for output_idx, sample_idx in enumerate(valid_indices):
                lead_record = lead_records[output_idx]
                lead_rows[sample_idx] = {
                    'seed_smiles': lead_record.seed_smiles,
                    'smiles': lead_record.smiles,
                    'reward': float(lead_record.reward),
                    'base_reward': -1.0 if not lead_record.is_valid else float(lead_record.reward) - config.sim_weight * float(lead_record.sim),
                    'sim': lead_record.sim,
                    'is_valid': bool(lead_record.is_valid),
                    'alert_hit': bool(lead_record.alert_hit),
                    'qed': lead_record.qed,
                    'sa': lead_record.sa,
                    'sa_score': lead_record.sa_score,
                    'soft_reward': lead_record.soft_reward,
                }

        rewards = [row['reward'] for row in lead_rows]
        base_rewards = [row['base_reward'] for row in lead_rows]
        sims = [row['sim'] for row in lead_rows]
        qeds = [row['qed'] for row in lead_rows]
        sas = [row['sa'] for row in lead_rows]
        sa_scores = [row['sa_score'] for row in lead_rows]
        soft_rewards = [row['soft_reward'] for row in lead_rows]
        valid_flags = [1.0 if row['is_valid'] else 0.0 for row in lead_rows]
        alert_flags = [1.0 if row['alert_hit'] else 0.0 for row in lead_rows]

        summary = {
            'experiment': experiment.name,
            'denovo_ckpt_path': experiment.denovo_ckpt_path,
            'lead_ckpt_path': experiment.lead_ckpt_path,
            'num_samples': len(lead_rows),
            'reward_mean': float(sum(rewards) / len(rewards)),
            'base_reward_mean': float(sum(base_rewards) / len(base_rewards)),
            'sim_mean': _nanmean(sims),
            'valid_fraction': float(sum(valid_flags) / len(valid_flags)),
            'alert_hit_fraction': float(sum(alert_flags) / len(alert_flags)),
            'qed_mean': _nanmean(qeds),
            'sa_mean': _nanmean(sas),
            'sa_score_mean': _nanmean(sa_scores),
            'soft_reward_mean': _nanmean(soft_rewards),
        }
        return summary, lead_rows
    finally:
        lead_reward.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    device = resolve_device(config.device)

    results = []
    all_rows = []
    for experiment in config.experiments:
        summary, rows = evaluate_experiment(config, experiment, device)
        results.append(summary)
        if config.output_rows_path is not None:
            for row in rows:
                all_rows.append({'experiment': experiment.name, **row})

    markdown = _build_markdown(results)
    _ensure_parent_dir(config.output_markdown_path)
    with open(config.output_markdown_path, 'w') as handle:
        handle.write(markdown)

    if config.output_json_path is not None:
        _ensure_parent_dir(config.output_json_path)
        with open(config.output_json_path, 'w') as handle:
            json.dump(results, handle, indent=2, sort_keys=True)

    if config.output_rows_path is not None:
        _ensure_parent_dir(config.output_rows_path)
        if os.path.exists(config.output_rows_path):
            os.remove(config.output_rows_path)
        for row in all_rows:
            write_jsonl(config.output_rows_path, row)

    logger.info('Wrote markdown results to %s', config.output_markdown_path)


if __name__ == '__main__':
    main()
