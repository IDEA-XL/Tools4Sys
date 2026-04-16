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
from genmol.mm.docking import CrossDockedDockingEvaluator, SUPPORTED_DOCKING_MODES
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
    crossdocked_root: str = ''
    docking_mode: str = 'vina_score'
    qvina_path: str | None = os.path.join(os.path.realpath('.'), 'scripts', 'exps', 'lead', 'docking', 'qvina02')
    docking_cache_dir: str | None = None
    docking_exhaustiveness: int = 8
    docking_num_cpu: int = 5
    docking_num_modes: int = 10
    docking_timeout_gen3d: int = 30
    docking_timeout_dock: int = 100
    docking_size_factor: float | None = 1.0
    docking_buffer: float = 5.0
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
    if not config.crossdocked_root:
        raise ValueError('crossdocked_root is required for docking evaluation')
    if not os.path.isdir(config.crossdocked_root):
        raise NotADirectoryError(f'crossdocked_root is not a directory: {config.crossdocked_root}')
    if config.docking_mode not in SUPPORTED_DOCKING_MODES:
        raise ValueError(f'docking_mode must be one of {SUPPORTED_DOCKING_MODES}, got {config.docking_mode!r}')
    if config.docking_mode == 'qvina':
        if not config.qvina_path:
            raise ValueError('qvina_path is required when docking_mode=qvina')
        if not os.path.exists(config.qvina_path):
            raise FileNotFoundError(f'qvina_path not found: {config.qvina_path}')
    if config.docking_exhaustiveness <= 0:
        raise ValueError('docking_exhaustiveness must be positive')
    if config.docking_num_cpu <= 0:
        raise ValueError('docking_num_cpu must be positive')
    if config.docking_num_modes <= 0:
        raise ValueError('docking_num_modes must be positive')
    if config.docking_size_factor is not None and config.docking_size_factor <= 0:
        raise ValueError('docking_size_factor must be positive when provided')
    if config.docking_buffer < 0:
        raise ValueError('docking_buffer must be non-negative')
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


def _docking_header_columns(config):
    if config.docking_mode == 'qvina':
        return ['QVina Mean', 'QVina Median', 'Docking Success']
    if config.docking_mode == 'vina_score':
        return ['Vina Score Mean', 'Vina Score Median', 'Vina Min Mean', 'Vina Min Median', 'Docking Success']
    return [
        'Vina Score Mean',
        'Vina Score Median',
        'Vina Min Mean',
        'Vina Min Median',
        'Vina Dock Mean',
        'Vina Dock Median',
        'Docking Success',
    ]


def _docking_row_values(config, row):
    if config.docking_mode == 'qvina':
        return [
            _format_metric(row['qvina_mean']),
            _format_metric(row['qvina_median']),
            _format_metric(row['docking_success_fraction']),
        ]
    if config.docking_mode == 'vina_score':
        return [
            _format_metric(row['vina_score_mean']),
            _format_metric(row['vina_score_median']),
            _format_metric(row['vina_min_mean']),
            _format_metric(row['vina_min_median']),
            _format_metric(row['docking_success_fraction']),
        ]
    return [
        _format_metric(row['vina_score_mean']),
        _format_metric(row['vina_score_median']),
        _format_metric(row['vina_min_mean']),
        _format_metric(row['vina_min_median']),
        _format_metric(row['vina_dock_mean']),
        _format_metric(row['vina_dock_median']),
        _format_metric(row['docking_success_fraction']),
    ]


def _build_markdown(config, results):
    docking_columns = _docking_header_columns(config)
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
        f'- `crossdocked_root`: `{config.crossdocked_root}`',
        f'- `docking_mode`: `{config.docking_mode}`',
        f'- `qvina_path`: `{config.qvina_path}`',
        f'- `docking_size_factor`: `{config.docking_size_factor}`',
        f'- `docking_buffer`: `{config.docking_buffer}`',
        '',
        '| '
        + ' | '.join(
            ['Model', 'Samples', *docking_columns, 'Official Validity', 'Official Uniqueness', 'Official Quality',
             'Official Diversity', 'QED', 'SA', 'Reward Mean', 'Alert Hit Rate', 'SA Score', 'Soft Reward']
        )
        + ' |',
        '| ' + ' | '.join(['---'] * (2 + len(docking_columns) + 10)) + ' |',
    ]
    for row in results:
        lines.append(
            '| '
            + ' | '.join(
                [
                    row['display_name'],
                    str(int(row['num_samples'])),
                    *_docking_row_values(config, row),
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
            '- Docking runs on one generated molecule per selected pocket.',
            '- Receptors are resolved with the same rule as TargetDiff: `dirname(ligand_filename) / (basename(ligand_filename)[:10] + ".pdb")` under `crossdocked_root`.',
            '- `Docking Success` is the fraction of samples whose docking run completed and returned the expected affinity fields.',
            '- Docking affinity means and medians are computed over successful dockings only, matching the official TargetDiff evaluation semantics.',
            '- Docking boxes follow the TargetDiff implementation: center and box size are derived from the generated ligand conformer, not from the native ligand geometry.',
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
        official_metrics, reward_metrics, reward_records, docking_metrics, docking_records = evaluation_kernel.summarize(
            rollout.smiles,
            entries=selected_entries,
        )
        rows = build_rows(selected_entries, specs, rollout, reward_records, docking_records=docking_records)
        summary = {
            'experiment': experiment.name,
            'display_name': _display_name(experiment),
            'checkpoint_path': experiment.checkpoint_path,
            'split': config.split,
            'num_samples': len(rows),
            **(docking_metrics or {}),
            **official_metrics,
            **reward_metrics,
        }
        if docking_metrics is not None and float(docking_metrics['docking_success_fraction']) <= 0.0:
            first_error = next(
                (row.get('docking_error') for row in rows if row.get('docking_error')),
                'unknown docking failure',
            )
            raise RuntimeError(
                'Docking evaluation produced zero successful dockings. '
                f'First docking error: {first_error}'
            )
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

    docking_cache_dir = config.docking_cache_dir
    if docking_cache_dir is None:
        json_root, _ = os.path.splitext(config.output_json_path)
        docking_cache_dir = json_root + '.docking_cache'

    evaluation_kernel = PocketPrefixEvaluationKernel(
        docking_model=CrossDockedDockingEvaluator(
            crossdocked_root=config.crossdocked_root,
            docking_mode=config.docking_mode,
            qvina_path=config.qvina_path,
            cache_dir=docking_cache_dir,
            exhaustiveness=config.docking_exhaustiveness,
            num_cpu_dock=config.docking_num_cpu,
            num_modes=config.docking_num_modes,
            timeout_gen3d=config.docking_timeout_gen3d,
            timeout_dock=config.docking_timeout_dock,
            size_factor=config.docking_size_factor,
            buffer=config.docking_buffer,
        )
    )
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
