# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass

import matplotlib
import torch
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.realpath('.'), 'src'))

from genmol.rl.policy import GenMolCpGRPOPolicy
from genmol.rl.reward import MolecularReward, compute_internal_diversity
from genmol.rl.specs import sample_group_specs
from genmol.rl.trainer import write_jsonl


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalExperimentConfig:
    name: str
    checkpoint_path: str
    display_name: str | None = None


@dataclass(frozen=True)
class EvalConfig:
    output_markdown_path: str
    output_json_path: str
    output_qed_diversity_plot_path: str
    output_sa_score_diversity_plot_path: str
    output_rows_path: str | None = None
    seed: int = 42
    bf16: bool = True
    device: str = 'cuda'
    num_samples: int = 1000
    generation_batch_size: int = 2048
    generation_temperature: float = 1.0
    randomness: float = 0.3
    randomness_values: list[float] | None = None
    min_add_len: int = 60
    max_completion_length: int | None = None
    length_path: str | None = None
    experiments: list[EvalExperimentConfig] | None = None


def configure_logging():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )


def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _resolve_randomness_values(raw_config):
    if 'randomness_values' in raw_config and raw_config['randomness_values'] is not None:
        return list(raw_config['randomness_values'])
    return [float(raw_config.get('randomness', 0.3))]


def load_config(path):
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    experiments = [EvalExperimentConfig(**item) for item in raw.pop('experiments')]
    randomness_values = _resolve_randomness_values(raw)
    raw.pop('randomness_values', None)
    config = EvalConfig(experiments=experiments, randomness_values=randomness_values, **raw)

    if config.num_samples <= 0:
        raise ValueError(f'num_samples must be positive, got {config.num_samples}')
    if config.generation_batch_size <= 0:
        raise ValueError(f'generation_batch_size must be positive, got {config.generation_batch_size}')
    if config.generation_temperature <= 0:
        raise ValueError(f'generation_temperature must be positive, got {config.generation_temperature}')
    if config.min_add_len <= 0:
        raise ValueError(f'min_add_len must be positive, got {config.min_add_len}')
    if config.randomness_values is None or not config.randomness_values:
        raise ValueError('randomness_values must be non-empty')
    for randomness in config.randomness_values:
        if float(randomness) <= 0:
            raise ValueError(f'randomness must be positive, got {randomness}')
    if len(set(float(value) for value in config.randomness_values)) != len(config.randomness_values):
        raise ValueError(f'randomness_values must be unique, got {config.randomness_values}')
    if config.experiments is None or not config.experiments:
        raise ValueError('experiments must be non-empty')
    experiment_names = [experiment.name for experiment in config.experiments]
    if len(set(experiment_names)) != len(experiment_names):
        raise ValueError(f'experiment names must be unique, got {experiment_names}')
    for experiment in config.experiments:
        if not experiment.name:
            raise ValueError('experiment name must be non-empty')
        if not os.path.exists(experiment.checkpoint_path):
            raise FileNotFoundError(f'checkpoint not found: {experiment.checkpoint_path}')
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


def _display_name(experiment):
    return experiment.display_name or experiment.name


def _plot_metric_tradeoff(results, experiments, randomness_values, x_key, x_label, y_key, y_label, title, output_path):
    experiment_order = {experiment.name: idx for idx, experiment in enumerate(experiments)}
    randomness_order = {float(value): idx for idx, value in enumerate(randomness_values)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for experiment in sorted(experiments, key=lambda item: experiment_order[item.name]):
        rows = [row for row in results if row['experiment'] == experiment.name]
        if len(rows) != len(randomness_values):
            raise ValueError(
                f'Expected {len(randomness_values)} rows for {experiment.name}, found {len(rows)}'
            )
        rows.sort(key=lambda row: randomness_order[float(row['randomness'])])
        x_values = []
        y_values = []
        for row in rows:
            x_value = float(row[x_key])
            y_value = float(row[y_key])
            if not math.isfinite(x_value):
                raise ValueError(
                    f'Non-finite {x_key} for experiment={experiment.name} randomness={row["randomness"]}: {x_value}'
                )
            if not math.isfinite(y_value):
                raise ValueError(
                    f'Non-finite {y_key} for experiment={experiment.name} randomness={row["randomness"]}: {y_value}'
                )
            x_values.append(x_value)
            y_values.append(y_value)

        ax.plot(x_values, y_values, marker='o', linewidth=2, label=_display_name(experiment))
        for row, x_value, y_value in zip(rows, x_values, y_values):
            ax.annotate(
                f'r={float(row["randomness"]):.1f}',
                (x_value, y_value),
                textcoords='offset points',
                xytext=(4, 4),
                fontsize=8,
            )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _build_markdown(config, results):
    lines = [
        '# De Novo SGRPO Evaluation',
        '',
        f'- `num_samples`: {config.num_samples}',
        f'- `generation_batch_size`: {config.generation_batch_size}',
        f'- `generation_temperature`: {config.generation_temperature}',
        f'- `min_add_len`: {config.min_add_len}',
        f'- `max_completion_length`: {config.max_completion_length}',
        f'- `randomness_values`: {", ".join(str(float(value)) for value in config.randomness_values)}',
        '',
        f'- `QED vs Diversity plot`: `{config.output_qed_diversity_plot_path}`',
        f'- `SA Score vs Diversity plot`: `{config.output_sa_score_diversity_plot_path}`',
        '',
        '| Model | Randomness | Overall De Novo Score | QED | SA Score | Soft Quality Score | Internal Diversity | Valid Molecule Rate | Alert Hit Rate | Invalid Rate |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in results:
        lines.append(
            '| '
            + ' | '.join(
                [
                    row['display_name'],
                    _format_metric(row['randomness']),
                    _format_metric(row['reward_mean']),
                    _format_metric(row['qed_mean']),
                    _format_metric(row['sa_score_mean']),
                    _format_metric(row['soft_reward_mean']),
                    _format_metric(row['diversity']),
                    _format_metric(row['valid_fraction']),
                    _format_metric(row['alert_hit_fraction']),
                    _format_metric(row['invalid_fraction']),
                ]
            )
            + ' |'
        )
    lines.extend(
        [
            '',
            'Column notes:',
            '- `Overall De Novo Score`: mean final molecule-level reward after invalid handling and alert gating.',
            '- `QED`: mean QED over valid generated molecules.',
            '- `SA Score`: mean bounded SA-derived score used by training. Higher is better.',
            '- `Soft Quality Score`: mean of `0.6 * QED + 0.4 * SA Score` over valid molecules.',
            '- `Internal Diversity`: `1 - mean(pairwise Tanimoto similarity)` computed over all generated valid molecules for that run.',
            '- `Valid Molecule Rate`: fraction of generated outputs that decode to valid molecules.',
            '- `Alert Hit Rate`: fraction of generated outputs that hit the alert rule set.',
            '- `Invalid Rate`: fraction of generated outputs that are invalid.',
            '',
            'Row notes:',
            '- Each row is one model evaluated at one randomness value.',
            '- The line plots connect rows for the same model in increasing randomness order.',
            '',
        ]
    )
    return '\n'.join(lines)


def evaluate_model(config, experiment, device, reward_model):
    logger.info('Evaluating %s', experiment.name)
    policy = GenMolCpGRPOPolicy(
        checkpoint_path=experiment.checkpoint_path,
        device=device,
        bf16=config.bf16,
        trainable=False,
    )

    results = []
    try:
        for randomness in config.randomness_values:
            logger.info('Evaluating %s at randomness=%s', experiment.name, randomness)
            group_specs = sample_group_specs(
                num_groups=config.num_samples,
                generation_temperature=config.generation_temperature,
                randomness=float(randomness),
                min_add_len=config.min_add_len,
                seed=config.seed,
                max_completion_length=config.max_completion_length,
                length_path=config.length_path,
            )
            rollout = policy.rollout_specs(
                specs=group_specs,
                generation_batch_size=min(config.generation_batch_size, len(group_specs)),
                seed=config.seed,
            )
            records = reward_model.score(rollout.smiles)
            if len(records) != config.num_samples:
                raise RuntimeError(
                    f'Reward record count mismatch for {experiment.name}: expected {config.num_samples}, got {len(records)}'
                )

            smiles = [record.smiles for record in records]
            rewards = [float(record.reward) for record in records]
            qeds = [record.qed for record in records]
            sas = [record.sa for record in records]
            sa_scores = [record.sa_score for record in records]
            soft_rewards = [record.soft_reward for record in records]
            valid_flags = [1.0 if record.is_valid else 0.0 for record in records]
            alert_flags = [1.0 if record.alert_hit else 0.0 for record in records]
            invalid_flags = [1.0 if not record.is_valid else 0.0 for record in records]

            results.append(
                {
                    'experiment': experiment.name,
                    'display_name': _display_name(experiment),
                    'checkpoint_path': experiment.checkpoint_path,
                    'num_samples': len(records),
                    'randomness': float(randomness),
                    'reward_mean': float(sum(rewards) / len(rewards)),
                    'qed_mean': _nanmean(qeds),
                    'sa_mean': _nanmean(sas),
                    'sa_score_mean': _nanmean(sa_scores),
                    'soft_reward_mean': _nanmean(soft_rewards),
                    'diversity': float(compute_internal_diversity(smiles)),
                    'valid_fraction': float(sum(valid_flags) / len(valid_flags)),
                    'alert_hit_fraction': float(sum(alert_flags) / len(alert_flags)),
                    'invalid_fraction': float(sum(invalid_flags) / len(invalid_flags)),
                }
            )

            if config.output_rows_path is not None:
                for idx, record in enumerate(records):
                    write_jsonl(
                        config.output_rows_path,
                        {
                            'experiment': experiment.name,
                            'display_name': _display_name(experiment),
                            'checkpoint_path': experiment.checkpoint_path,
                            'sample_index': idx,
                            'randomness': float(randomness),
                            'reward': float(record.reward),
                            'is_valid': bool(record.is_valid),
                            'alert_hit': bool(record.alert_hit),
                            'qed': record.qed,
                            'sa': record.sa,
                            'sa_score': record.sa_score,
                            'soft_reward': record.soft_reward,
                            'smiles': record.smiles,
                        },
                    )
    finally:
        del policy
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    device = resolve_device(config.device)

    if config.output_rows_path is not None:
        _ensure_parent_dir(config.output_rows_path)
        if os.path.exists(config.output_rows_path):
            os.remove(config.output_rows_path)

    reward_model = MolecularReward()
    try:
        results = []
        for experiment in config.experiments:
            results.extend(evaluate_model(config, experiment, device, reward_model))
    finally:
        reward_model.close()

    experiment_order = {experiment.name: idx for idx, experiment in enumerate(config.experiments)}
    randomness_order = {float(value): idx for idx, value in enumerate(config.randomness_values)}
    results.sort(key=lambda row: (experiment_order[row['experiment']], randomness_order[float(row['randomness'])]))

    _plot_metric_tradeoff(
        results=results,
        experiments=config.experiments,
        randomness_values=config.randomness_values,
        x_key='qed_mean',
        x_label='QED',
        y_key='diversity',
        y_label='Internal Diversity',
        title='QED vs Internal Diversity',
        output_path=config.output_qed_diversity_plot_path,
    )
    _plot_metric_tradeoff(
        results=results,
        experiments=config.experiments,
        randomness_values=config.randomness_values,
        x_key='sa_score_mean',
        x_label='SA Score',
        y_key='diversity',
        y_label='Internal Diversity',
        title='SA Score vs Internal Diversity',
        output_path=config.output_sa_score_diversity_plot_path,
    )

    markdown = _build_markdown(config, results)
    _ensure_parent_dir(config.output_markdown_path)
    with open(config.output_markdown_path, 'w') as handle:
        handle.write(markdown)

    _ensure_parent_dir(config.output_json_path)
    with open(config.output_json_path, 'w') as handle:
        json.dump(results, handle, indent=2, sort_keys=True)

    logger.info('Wrote markdown results to %s', config.output_markdown_path)
    logger.info('Wrote JSON results to %s', config.output_json_path)
    logger.info('Wrote QED-vs-diversity plot to %s', config.output_qed_diversity_plot_path)
    logger.info('Wrote SA-score-vs-diversity plot to %s', config.output_sa_score_diversity_plot_path)


if __name__ == '__main__':
    main()
