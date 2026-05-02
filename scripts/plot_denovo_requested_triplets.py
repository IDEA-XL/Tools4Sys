#!/usr/bin/env python3
"""Plot requested de novo three-model Pareto figures into genmol/figs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = REPO_ROOT / "sgrpo-main-results" / "genmol-denovo"
OUTPUT_DIR = REPO_ROOT / "figs"

RANDOMNESS_SUMMARY = SUMMARY_DIR / "denovo_main_results_randomness_sweep_20260425.json"
TEMPERATURE_SUMMARY = SUMMARY_DIR / "denovo_main_results_temperature_sweep_20260425.json"

MARKER_CYCLE = ("o", "^", "s")


@dataclass(frozen=True)
class PlotSpec:
    output_name: str
    summary_path: Path
    metric_key: str
    metric_label: str
    experiments: tuple[str, ...]
    title: str


EXPERIMENT_LABELS = {
    "original_genmol_v2": "Original GenMol v2",
    "genmol_denovo_grpo": "GenMol De Novo GRPO 1000",
    "genmol_denovo_sgrpo_rewardsum_loo_1000": "GenMol De Novo SGRPO RewardSum LOO 1000",
    "genmol_denovo_grpo_2000": "GenMol De Novo GRPO 2000",
    "genmol_denovo_sgrpo_rewardsum_loo_2000": "GenMol De Novo SGRPO RewardSum LOO 2000",
}

METRICS = (
    ("qed_mean", "QED"),
    ("sa_score_mean", "SA Score"),
    ("soft_reward_mean", "Soft Quality Score"),
)

PLOT_SPECS = (
    PlotSpec(
        output_name="denovo_qed_diversity_randomness_1000.png",
        summary_path=RANDOMNESS_SUMMARY,
        metric_key="qed_mean",
        metric_label="QED",
        experiments=(
            "original_genmol_v2",
            "genmol_denovo_grpo",
            "genmol_denovo_sgrpo_rewardsum_loo_1000",
        ),
        title="QED vs Internal Diversity (Randomness Sweep, 1000-Step Models)",
    ),
    PlotSpec(
        output_name="denovo_sa_score_diversity_randomness_1000.png",
        summary_path=RANDOMNESS_SUMMARY,
        metric_key="sa_score_mean",
        metric_label="SA Score",
        experiments=(
            "original_genmol_v2",
            "genmol_denovo_grpo",
            "genmol_denovo_sgrpo_rewardsum_loo_1000",
        ),
        title="SA Score vs Internal Diversity (Randomness Sweep, 1000-Step Models)",
    ),
    PlotSpec(
        output_name="denovo_soft_quality_diversity_randomness_1000.png",
        summary_path=RANDOMNESS_SUMMARY,
        metric_key="soft_reward_mean",
        metric_label="Soft Quality Score",
        experiments=(
            "original_genmol_v2",
            "genmol_denovo_grpo",
            "genmol_denovo_sgrpo_rewardsum_loo_1000",
        ),
        title="Soft Quality Score vs Internal Diversity (Randomness Sweep, 1000-Step Models)",
    ),
    PlotSpec(
        output_name="denovo_qed_diversity_temperature_1000.png",
        summary_path=TEMPERATURE_SUMMARY,
        metric_key="qed_mean",
        metric_label="QED",
        experiments=(
            "original_genmol_v2",
            "genmol_denovo_grpo",
            "genmol_denovo_sgrpo_rewardsum_loo_1000",
        ),
        title="QED vs Internal Diversity (Temperature Sweep, 1000-Step Models)",
    ),
    PlotSpec(
        output_name="denovo_sa_score_diversity_temperature_1000.png",
        summary_path=TEMPERATURE_SUMMARY,
        metric_key="sa_score_mean",
        metric_label="SA Score",
        experiments=(
            "original_genmol_v2",
            "genmol_denovo_grpo",
            "genmol_denovo_sgrpo_rewardsum_loo_1000",
        ),
        title="SA Score vs Internal Diversity (Temperature Sweep, 1000-Step Models)",
    ),
    PlotSpec(
        output_name="denovo_soft_quality_diversity_temperature_1000.png",
        summary_path=TEMPERATURE_SUMMARY,
        metric_key="soft_reward_mean",
        metric_label="Soft Quality Score",
        experiments=(
            "original_genmol_v2",
            "genmol_denovo_grpo",
            "genmol_denovo_sgrpo_rewardsum_loo_1000",
        ),
        title="Soft Quality Score vs Internal Diversity (Temperature Sweep, 1000-Step Models)",
    ),
    PlotSpec(
        output_name="denovo_qed_diversity_randomness_2000.png",
        summary_path=RANDOMNESS_SUMMARY,
        metric_key="qed_mean",
        metric_label="QED",
        experiments=(
            "original_genmol_v2",
            "genmol_denovo_grpo_2000",
            "genmol_denovo_sgrpo_rewardsum_loo_2000",
        ),
        title="QED vs Internal Diversity (Randomness Sweep, 2000-Step Models)",
    ),
    PlotSpec(
        output_name="denovo_sa_score_diversity_randomness_2000.png",
        summary_path=RANDOMNESS_SUMMARY,
        metric_key="sa_score_mean",
        metric_label="SA Score",
        experiments=(
            "original_genmol_v2",
            "genmol_denovo_grpo_2000",
            "genmol_denovo_sgrpo_rewardsum_loo_2000",
        ),
        title="SA Score vs Internal Diversity (Randomness Sweep, 2000-Step Models)",
    ),
    PlotSpec(
        output_name="denovo_soft_quality_diversity_randomness_2000.png",
        summary_path=RANDOMNESS_SUMMARY,
        metric_key="soft_reward_mean",
        metric_label="Soft Quality Score",
        experiments=(
            "original_genmol_v2",
            "genmol_denovo_grpo_2000",
            "genmol_denovo_sgrpo_rewardsum_loo_2000",
        ),
        title="Soft Quality Score vs Internal Diversity (Randomness Sweep, 2000-Step Models)",
    ),
    PlotSpec(
        output_name="denovo_qed_diversity_temperature_2000.png",
        summary_path=TEMPERATURE_SUMMARY,
        metric_key="qed_mean",
        metric_label="QED",
        experiments=(
            "original_genmol_v2",
            "genmol_denovo_grpo_2000",
            "genmol_denovo_sgrpo_rewardsum_loo_2000",
        ),
        title="QED vs Internal Diversity (Temperature Sweep, 2000-Step Models)",
    ),
    PlotSpec(
        output_name="denovo_sa_score_diversity_temperature_2000.png",
        summary_path=TEMPERATURE_SUMMARY,
        metric_key="sa_score_mean",
        metric_label="SA Score",
        experiments=(
            "original_genmol_v2",
            "genmol_denovo_grpo_2000",
            "genmol_denovo_sgrpo_rewardsum_loo_2000",
        ),
        title="SA Score vs Internal Diversity (Temperature Sweep, 2000-Step Models)",
    ),
    PlotSpec(
        output_name="denovo_soft_quality_diversity_temperature_2000.png",
        summary_path=TEMPERATURE_SUMMARY,
        metric_key="soft_reward_mean",
        metric_label="Soft Quality Score",
        experiments=(
            "original_genmol_v2",
            "genmol_denovo_grpo_2000",
            "genmol_denovo_sgrpo_rewardsum_loo_2000",
        ),
        title="Soft Quality Score vs Internal Diversity (Temperature Sweep, 2000-Step Models)",
    ),
)


def load_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise TypeError(f"Expected JSON list in {path}, got {type(rows).__name__}")
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def require_finite(row: dict, key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(
            f"Non-finite value for {key}: experiment={row['experiment']} "
            f"sweep_axis={row['sweep_axis']} sweep_value={row['sweep_value']} value={value}"
        )
    return value


def validate_rows(rows: list[dict], experiments: tuple[str, ...]) -> tuple[str, list[float], dict[tuple[str, float], dict]]:
    sweep_axes = {row["sweep_axis"] for row in rows}
    if len(sweep_axes) != 1:
        raise ValueError(f"Expected exactly one sweep axis, got {sorted(sweep_axes)}")
    sweep_axis = next(iter(sweep_axes))
    row_index: dict[tuple[str, float], dict] = {}
    for row in rows:
        key = (row["experiment"], float(row["sweep_value"]))
        if key in row_index:
            raise ValueError(f"Duplicate row for experiment={key[0]} sweep_value={key[1]}")
        row_index[key] = row
    sweep_values = sorted({float(row["sweep_value"]) for row in rows if row["experiment"] == experiments[0]})
    if not sweep_values:
        raise ValueError(f"No sweep values found for anchor experiment {experiments[0]}")
    for experiment in experiments:
        for sweep_value in sweep_values:
            if (experiment, sweep_value) not in row_index:
                raise ValueError(
                    f"Missing row for experiment={experiment} sweep_axis={sweep_axis} sweep_value={sweep_value}"
                )
    return sweep_axis, sweep_values, row_index


def plot_spec(spec: PlotSpec) -> Path:
    rows = load_rows(spec.summary_path)
    sweep_axis, sweep_values, row_index = validate_rows(rows, spec.experiments)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color")
    if not color_cycle or len(color_cycle) < len(spec.experiments):
        raise ValueError("Matplotlib color cycle is unexpectedly short")

    for series_idx, experiment in enumerate(spec.experiments):
        experiment_rows = [row_index[(experiment, sweep_value)] for sweep_value in sweep_values]
        x_values = [require_finite(row, spec.metric_key) for row in experiment_rows]
        y_values = [require_finite(row, "diversity") for row in experiment_rows]
        label = EXPERIMENT_LABELS[experiment]
        ax.plot(
            x_values,
            y_values,
            color=color_cycle[series_idx],
            marker=MARKER_CYCLE[series_idx],
            linewidth=2.2,
            markersize=6,
            label=label,
        )
        for row, x_value, y_value in zip(experiment_rows, x_values, y_values):
            ax.annotate(
                f"{float(row['sweep_value']):.1f}",
                (x_value, y_value),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )

    ax.set_title(spec.title)
    ax.set_xlabel(spec.metric_label)
    ax.set_ylabel("Internal Diversity")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / spec.output_name
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main() -> None:
    missing_experiments = [name for name in EXPERIMENT_LABELS if name is None]
    if missing_experiments:
        raise ValueError(f"Unexpected missing experiment labels: {missing_experiments}")

    generated = [plot_spec(spec) for spec in PLOT_SPECS]
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
