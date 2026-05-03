#!/usr/bin/env python3
"""Plot the de novo 2000-step paired-sweep ablation figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = (
    REPO_ROOT
    / "sgrpo-main-results"
    / "genmol-denovo"
    / "denovo_main_results_paired_sweep_20260427.json"
)
OUTPUT_PATH = REPO_ROOT / "figs" / "ablation.pdf"

EXPERIMENT_SPECS = (
    ("genmol_denovo_sgrpo_rewardsum_loo_2000", "SGRPO", "#1F4E79", "o"),
    ("genmol_denovo_sgrpo_rewardsum_2000", "SGRPO w.o. LOO", "#D97706", "s"),
    ("genmol_denovo_grpo_2000", "GRPO", "#2A9D8F", "^"),
)

EXPECTED_SWEEP_LABELS = (
    "r=0.1,t=0.5",
    "r=0.3,t=0.8",
    "r=0.5,t=1.1",
    "r=0.7,t=1.4",
    "r=0.9,t=1.7",
    "r=1.0,t=2.0",
)


def load_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise TypeError(f"Expected a JSON list in {path}, got {type(rows).__name__}")
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def build_experiment_rows(rows: list[dict], experiment: str) -> list[dict]:
    subset = [row for row in rows if row["experiment"] == experiment]
    if not subset:
        raise ValueError(f"Missing experiment {experiment} in {RESULTS_PATH}")
    if {row["sweep_axis"] for row in subset} != {"randomness_temperature_pair"}:
        raise ValueError(f"Unexpected sweep axis for {experiment}")
    row_by_label = {str(row["sweep_label"]): row for row in subset}
    missing_labels = [label for label in EXPECTED_SWEEP_LABELS if label not in row_by_label]
    if missing_labels:
        raise ValueError(f"Missing sweep labels for {experiment}: {missing_labels}")
    return [row_by_label[label] for label in EXPECTED_SWEEP_LABELS]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.labelsize": 21,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 5.5,
            "ytick.major.size": 5.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def add_terminal_arrow(ax: plt.Axes, x_values: list[float], y_values: list[float], color: str) -> None:
    ax.annotate(
        "",
        xy=(x_values[-1], y_values[-1]),
        xytext=(x_values[-2], y_values[-2]),
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 2.2,
            "color": color,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
            "mutation_scale": 14,
            "alpha": 0.95,
        },
        zorder=4,
    )


def mean_point(points: list[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        raise ValueError("Expected at least one point")
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def plot() -> Path:
    configure_style()
    rows = load_rows(RESULTS_PATH)

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAF8")

    all_x: list[float] = []
    all_y: list[float] = []
    low_endpoints: list[tuple[float, float]] = []
    high_endpoints: list[tuple[float, float]] = []

    for experiment, legend_label, color, marker in EXPERIMENT_SPECS:
        series_rows = build_experiment_rows(rows, experiment)
        x_values = [float(row["soft_reward_mean"]) for row in series_rows]
        y_values = [float(row["diversity"]) for row in series_rows]
        all_x.extend(x_values)
        all_y.extend(y_values)
        low_endpoints.append((x_values[0], y_values[0]))
        high_endpoints.append((x_values[-1], y_values[-1]))

        ax.plot(
            x_values,
            y_values,
            color=color,
            linewidth=2.8,
            marker=marker,
            markersize=8.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=1.2,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=legend_label,
            zorder=3,
        )
        add_terminal_arrow(ax, x_values, y_values, color)

    ax.set_xlabel("Utility")
    ax.set_ylabel("Diversity")
    ax.grid(True, color="#D8D8D8", linewidth=0.9, alpha=0.45)

    x_margin = (max(all_x) - min(all_x)) * 0.08
    y_margin = (max(all_y) - min(all_y)) * 0.10
    ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin * 0.65)
    ax.set_ylim(min(all_y) - y_margin * 0.55, max(all_y) + y_margin * 0.55)

    ax.legend(
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#D9D9D9",
        framealpha=0.92,
        borderpad=0.6,
        handlelength=2.2,
        handletextpad=0.6,
    )

    low_x, low_y = mean_point(low_endpoints)
    high_x, high_y = mean_point(high_endpoints)
    ax.annotate(
        "Low Rand. & Temp.",
        xy=(low_x, low_y),
        xytext=(0.775, 0.550),
        ha="left",
        va="center",
        fontsize=14,
        color="#555555",
        zorder=5,
    )
    ax.annotate(
        "High Rand. & Temp.",
        xy=(high_x, high_y),
        xytext=(0.662, 0.856),
        ha="left",
        va="top",
        fontsize=14,
        color="#555555",
        zorder=5,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)
    return OUTPUT_PATH


if __name__ == "__main__":
    output_path = plot()
    print(output_path)
