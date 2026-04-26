#!/usr/bin/env python
"""Plot split de novo Pareto curves from existing sweep summary JSON files."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PlotGroup:
    suffix: str
    title_suffix: str
    experiments: tuple[str, ...]


PLOT_GROUPS = (
    PlotGroup(
        suffix='1000',
        title_suffix='1000-step models',
        experiments=(
            'original_genmol_v2',
            'genmol_denovo_grpo',
            'genmol_denovo_grpo_q08_sa02_1000',
            'genmol_denovo_sgrpo',
            'genmol_denovo_sgrpo_thr_q085_sa072_1000',
            'genmol_denovo_sgrpo_rewardsum_1000',
            'genmol_denovo_sgrpo_thr_q085_sa072_rewardsum_1000',
            'genmol_denovo_sgrpo_hierarchicalsum_1000',
            'genmol_denovo_sgrpo_rewardsum_loo_1000',
            'genmol_denovo_sgrpo_gw05_rewardsum_loo_1000',
            'genmol_denovo_sgrpo_rewardsum_tempsamp_rndsamp_1000',
            'genmol_denovo_sgrpo_rewardsum_loo_tempsamp_rndsamp_1000',
            'genmol_denovo_sgrpo_gw05_rewardsum_loo_q08_sa02_1000',
        ),
    ),
    PlotGroup(
        suffix='2000',
        title_suffix='2000-step models',
        experiments=(
            'original_genmol_v2',
            'genmol_denovo_grpo_2000',
            'genmol_denovo_grpo_q08_sa02_2000',
            'genmol_denovo_sgrpo_2000',
            'genmol_denovo_grpo_divreg005_2000',
            'genmol_denovo_sgrpo_thr_q085_sa072_2000',
            'genmol_denovo_sgrpo_rewardsum_2000',
            'genmol_denovo_sgrpo_thr_q085_sa072_rewardsum_2000',
            'genmol_denovo_sgrpo_hierarchicalsum_2000',
            'genmol_denovo_sgrpo_rewardsum_loo_2000',
            'genmol_denovo_sgrpo_gw05_rewardsum_loo_2000',
            'genmol_denovo_sgrpo_rewardsum_tempsamp_rndsamp_2000',
            'genmol_denovo_sgrpo_rewardsum_loo_tempsamp_rndsamp_2000',
            'genmol_denovo_sgrpo_gw05_rewardsum_loo_q08_sa02_2000',
        ),
    ),
)

LEGACY_SOFT_REWARD_PLOT_GROUPS = (
    PlotGroup(
        suffix='1000',
        title_suffix='1000-step legacy-weight models',
        experiments=(
            'original_genmol_v2',
            'genmol_denovo_grpo',
            'genmol_denovo_sgrpo',
            'genmol_denovo_sgrpo_thr_q085_sa072_1000',
            'genmol_denovo_sgrpo_rewardsum_1000',
            'genmol_denovo_sgrpo_thr_q085_sa072_rewardsum_1000',
            'genmol_denovo_sgrpo_hierarchicalsum_1000',
            'genmol_denovo_sgrpo_rewardsum_loo_1000',
            'genmol_denovo_sgrpo_rewardsum_tempsamp_rndsamp_1000',
            'genmol_denovo_sgrpo_rewardsum_loo_tempsamp_rndsamp_1000',
        ),
    ),
    PlotGroup(
        suffix='2000',
        title_suffix='2000-step legacy-weight models',
        experiments=(
            'original_genmol_v2',
            'genmol_denovo_grpo_2000',
            'genmol_denovo_sgrpo_2000',
            'genmol_denovo_grpo_divreg005_2000',
            'genmol_denovo_sgrpo_thr_q085_sa072_2000',
            'genmol_denovo_sgrpo_rewardsum_2000',
            'genmol_denovo_sgrpo_thr_q085_sa072_rewardsum_2000',
            'genmol_denovo_sgrpo_hierarchicalsum_2000',
            'genmol_denovo_sgrpo_rewardsum_loo_2000',
            'genmol_denovo_sgrpo_rewardsum_tempsamp_rndsamp_2000',
            'genmol_denovo_sgrpo_rewardsum_loo_tempsamp_rndsamp_2000',
        ),
    ),
)

NEW_VARIANT_SOFT_REWARD_PLOT_GROUPS = (
    PlotGroup(
        suffix='new_1000',
        title_suffix='1000-step new variants',
        experiments=(
            'genmol_denovo_grpo_q08_sa02_1000',
            'genmol_denovo_sgrpo_gw05_rewardsum_loo_1000',
            'genmol_denovo_sgrpo_gw05_rewardsum_loo_q08_sa02_1000',
        ),
    ),
    PlotGroup(
        suffix='new_2000',
        title_suffix='2000-step new variants',
        experiments=(
            'genmol_denovo_grpo_q08_sa02_2000',
            'genmol_denovo_sgrpo_gw05_rewardsum_loo_2000',
            'genmol_denovo_sgrpo_gw05_rewardsum_loo_q08_sa02_2000',
        ),
    ),
)

MAIN_METRICS = (
    ('qed_mean', 'QED', 'qed'),
    ('sa_score_mean', 'SA Score', 'sa_score'),
)

LEGACY_SOFT_REWARD_METRIC = ('soft_reward_mean', 'Soft Quality Score', 'soft_reward')
NEW_VARIANT_SOFT_REWARD_METRIC = ('soft_reward_mean', 'Soft Quality Score', 'soft_reward_new_variants')


def _load_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise TypeError(f'Expected a JSON list in {path}, got {type(rows).__name__}')
    if not rows:
        raise ValueError(f'No rows found in {path}')
    return rows


def _format_sweep_value(value: float) -> str:
    return f'{value:.1f}'


def _require_finite(row: dict, key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(
            f'Non-finite {key} for experiment={row["experiment"]} '
            f'{row["sweep_axis"]}={row["sweep_value"]}: {value}'
        )
    return value


def _validate_sweep(rows: Iterable[dict]) -> tuple[str, list[float]]:
    axes = {row['sweep_axis'] for row in rows}
    if len(axes) != 1:
        raise ValueError(f'Expected one sweep axis, found {sorted(axes)}')
    axis = next(iter(axes))
    values = sorted({float(row['sweep_value']) for row in rows})
    if not values:
        raise ValueError('No sweep values found')
    return axis, values


def _plot_group(rows: list[dict], group: PlotGroup, metric_key: str, metric_label: str, output_path: Path) -> None:
    sweep_axis, sweep_values = _validate_sweep(rows)
    row_index: dict[tuple[str, float], dict] = {}
    for row in rows:
        key = (row['experiment'], float(row['sweep_value']))
        if key in row_index:
            raise ValueError(f'Duplicate row for experiment={key[0]} {sweep_axis}={key[1]}')
        row_index[key] = row

    missing = [
        (experiment, value)
        for experiment in group.experiments
        for value in sweep_values
        if (experiment, value) not in row_index
    ]
    if missing:
        preview = ', '.join(f'{name}@{value}' for name, value in missing[:10])
        raise ValueError(f'Missing {len(missing)} required rows for {group.suffix}: {preview}')

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for experiment in group.experiments:
        experiment_rows = [row_index[(experiment, value)] for value in sweep_values]
        x_values = [_require_finite(row, metric_key) for row in experiment_rows]
        y_values = [_require_finite(row, 'diversity') for row in experiment_rows]
        display_name = str(experiment_rows[0].get('display_name') or experiment)
        ax.plot(x_values, y_values, marker='o', linewidth=2, label=display_name)
        for row, x_value, y_value in zip(experiment_rows, x_values, y_values):
            ax.annotate(
                _format_sweep_value(float(row['sweep_value'])),
                (x_value, y_value),
                textcoords='offset points',
                xytext=(4, 4),
                fontsize=7,
            )

    ax.set_title(f'{metric_label} vs Internal Diversity ({group.title_suffix})')
    ax.set_xlabel(metric_label)
    ax.set_ylabel('Internal Diversity')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_split_summary(summary_path: Path, output_dir: Path, name_prefix: str) -> list[Path]:
    rows = _load_rows(summary_path)
    output_paths = []
    for metric_key, metric_label, metric_name in MAIN_METRICS:
        for group in PLOT_GROUPS:
            output_path = output_dir / f'{metric_name}_vs_diversity_{name_prefix}_{group.suffix}_20260425.png'
            _plot_group(rows, group, metric_key, metric_label, output_path)
            output_paths.append(output_path)
    metric_key, metric_label, metric_name = LEGACY_SOFT_REWARD_METRIC
    for group in LEGACY_SOFT_REWARD_PLOT_GROUPS:
        output_path = output_dir / f'{metric_name}_vs_diversity_{name_prefix}_{group.suffix}_20260425.png'
        _plot_group(rows, group, metric_key, metric_label, output_path)
        output_paths.append(output_path)
    metric_key, metric_label, metric_name = NEW_VARIANT_SOFT_REWARD_METRIC
    for group in NEW_VARIANT_SOFT_REWARD_PLOT_GROUPS:
        output_path = output_dir / f'{metric_name}_vs_diversity_{name_prefix}_{group.suffix}_20260427.png'
        _plot_group(rows, group, metric_key, metric_label, output_path)
        output_paths.append(output_path)
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--randomness-json', type=Path, required=True)
    parser.add_argument('--temperature-json', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    outputs = []
    outputs.extend(_plot_split_summary(args.randomness_json, args.output_dir, 'randomness'))
    outputs.extend(_plot_split_summary(args.temperature_json, args.output_dir, 'temperature'))
    for output in outputs:
        print(output)


if __name__ == '__main__':
    main()
