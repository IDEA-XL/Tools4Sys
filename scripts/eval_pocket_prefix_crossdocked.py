import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass

import torch
import yaml

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.realpath('.'), 'src'))

from genmol.mm.crossdocked import load_crossdocked_manifest
from genmol.mm.evaluation import PocketPrefixEvaluationKernel, build_rows, select_manifest_entries
from genmol.mm.policy import PocketPrefixCpGRPOPolicy
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
    output_rows_path: str | None = None
    seed: int = 42
    bf16: bool = True
    device: str = 'cuda'
    manifest_path: str = ''
    split: str = 'test'
    num_pockets: int | None = None
    generation_batch_size: int = 1024
    generation_temperature: float = 1.0
    randomness: float = 0.3
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


def _format_metric(value):
    if value is None:
        return 'nan'
    value = float(value)
    if math.isnan(value):
        return 'nan'
    return f'{value:.6f}'


def load_config(path):
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    experiments = [EvalExperimentConfig(**item) for item in raw.pop('experiments')]
    config = EvalConfig(experiments=experiments, **raw)
    if not config.manifest_path:
        raise ValueError('manifest_path is required')
    if not os.path.exists(config.manifest_path):
        raise FileNotFoundError(f'manifest not found: {config.manifest_path}')
    if config.split not in {'train', 'val', 'test'}:
        raise ValueError(f"split must be one of 'train', 'val', 'test', got {config.split!r}")
    if config.generation_batch_size <= 0:
        raise ValueError('generation_batch_size must be positive')
    if config.generation_temperature <= 0:
        raise ValueError('generation_temperature must be positive')
    if config.randomness <= 0:
        raise ValueError('randomness must be positive')
    if config.min_add_len <= 0:
        raise ValueError('min_add_len must be positive')
    if config.experiments is None or not config.experiments:
        raise ValueError('experiments must be non-empty')
    for experiment in config.experiments:
        if not os.path.exists(experiment.checkpoint_path):
            raise FileNotFoundError(f'checkpoint not found: {experiment.checkpoint_path}')
    return config


def resolve_device(device_name):
    if device_name == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('device=cuda requested but CUDA is not available')
        return torch.device('cuda')
    return torch.device(device_name)


def _display_name(experiment):
    return experiment.display_name or experiment.name


def _build_markdown(config, results):
    lines = [
        '# Pocket Prefix CrossDocked Evaluation',
        '',
        f'- `manifest_path`: `{config.manifest_path}`',
        f'- `split`: `{config.split}`',
        f'- `num_pockets`: `{config.num_pockets if config.num_pockets is not None else "all"}`',
        f'- `generation_batch_size`: `{config.generation_batch_size}`',
        f'- `generation_temperature`: `{config.generation_temperature}`',
        f'- `randomness`: `{config.randomness}`',
        f'- `min_add_len`: `{config.min_add_len}`',
        f'- `max_completion_length`: `{config.max_completion_length}`',
        '',
        '| Model | Samples | Official Validity | Official Uniqueness | Official Quality | Official Diversity | QED | SA | Reward Mean | Alert Hit Rate | SA Score | Soft Reward |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in results:
        lines.append(
            '| '
            + ' | '.join(
                [
                    row['display_name'],
                    str(int(row['num_samples'])),
                    _format_metric(row['official_validity']),
                    _format_metric(row['official_uniqueness']),
                    _format_metric(row['official_quality']),
                    _format_metric(row['official_diversity']),
                    _format_metric(row['official_qed_mean']),
                    _format_metric(row['official_sa_mean']),
                    _format_metric(row['reward_mean']),
                    _format_metric(row['alert_hit_fraction']),
                    _format_metric(row['sa_score_mean']),
                    _format_metric(row['soft_reward_mean']),
                ]
            )
            + ' |'
        )
    lines.extend(
        [
            '',
            'Column notes:',
            '- `Official Validity`, `Official Uniqueness`, `Official Quality`, and `Official Diversity` match the implementations used in the official GenMol de novo / fragment-constrained scripts.',
            '- `Official Quality` is the fraction of generated outputs that are valid, unique, satisfy `QED >= 0.6`, and satisfy `SA <= 4`.',
            '- `QED` and `SA` are the means over the unique valid set used by the official metric computation.',
            '- `Reward Mean`, `Alert Hit Rate`, `SA Score`, and `Soft Reward` are auxiliary diagnostics from the current training reward implementation.',
            '',
        ]
    )
    return '\n'.join(lines)


def evaluate_experiment(config, experiment, device, selected_entries, specs, evaluation_kernel):
    logger.info('Evaluating %s on %d pockets', experiment.name, len(selected_entries))
    policy = PocketPrefixCpGRPOPolicy(
        checkpoint_path=experiment.checkpoint_path,
        device=device,
        bf16=config.bf16,
        trainable=False,
    )
    try:
        pocket_raw_embeddings, pocket_mask = policy.get_pocket_raw_embeddings(
            [entry['pocket_coords'] for entry in selected_entries]
        )
        rollout = policy.rollout_specs(
            specs=specs,
            pocket_raw_embeddings=pocket_raw_embeddings,
            pocket_mask=pocket_mask,
            generation_batch_size=min(config.generation_batch_size, len(selected_entries)),
            seed=config.seed,
        )
        official_metrics, reward_metrics, reward_records = evaluation_kernel.summarize(rollout.smiles)
        rows = build_rows(selected_entries, specs, rollout, reward_records)
        summary = {
            'experiment': experiment.name,
            'display_name': _display_name(experiment),
            'checkpoint_path': experiment.checkpoint_path,
            'split': config.split,
            'num_samples': len(rows),
            **official_metrics,
            **reward_metrics,
        }
        return summary, rows
    finally:
        del policy
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    device = resolve_device(config.device)

    entries, _ = load_crossdocked_manifest(config.manifest_path, config.split)
    selected_entries = select_manifest_entries(entries, config.num_pockets, config.seed)
    specs = sample_group_specs(
        num_groups=len(selected_entries),
        generation_temperature=config.generation_temperature,
        randomness=config.randomness,
        min_add_len=config.min_add_len,
        seed=config.seed,
        max_completion_length=config.max_completion_length,
        length_path=config.length_path,
    )

    evaluation_kernel = PocketPrefixEvaluationKernel()
    try:
        results = []
        all_rows = []
        for experiment in config.experiments:
            summary, rows = evaluate_experiment(
                config=config,
                experiment=experiment,
                device=device,
                selected_entries=selected_entries,
                specs=specs,
                evaluation_kernel=evaluation_kernel,
            )
            results.append(summary)
            if config.output_rows_path is not None:
                for row in rows:
                    all_rows.append({'experiment': experiment.name, 'display_name': _display_name(experiment), **row})
    finally:
        evaluation_kernel.close()

    markdown = _build_markdown(config, results)
    _ensure_parent_dir(config.output_markdown_path)
    with open(config.output_markdown_path, 'w') as handle:
        handle.write(markdown)

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
    logger.info('Wrote JSON results to %s', config.output_json_path)
    if config.output_rows_path is not None:
        logger.info('Wrote per-sample rows to %s', config.output_rows_path)


if __name__ == '__main__':
    main()
