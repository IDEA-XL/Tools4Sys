#!/usr/bin/env python3
"""Plot hyperparameter ablations using HV on the de novo paired sweep."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main_pareto_common import DE_NOVO_RESULTS_PATH, load_json


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "figs" / "hyperparam.pdf"

GROUP_SIZE_COLOR = "#8E63B6"
DIVERSITY_WEIGHT_COLOR = "#D1495B"

EXPECTED_PAIRED_SWEEP_LABELS = (
    "r=1.0,t=2.0",
    "r=0.9,t=1.7",
    "r=0.7,t=1.4",
    "r=0.5,t=1.1",
    "r=0.3,t=0.8",
    "r=0.1,t=0.5",
)


@dataclass(frozen=True)
class Point:
    utility: float
    diversity: float


@dataclass(frozen=True)
class CurveSpec:
    x_value: float
    experiment: str


GROUP_SIZE_SPECS = (
    CurveSpec(1, "genmol_denovo_grpo_2000"),
    CurveSpec(4, "genmol_denovo_sgrpo_ng4_rewardsum_loo_2000"),
    CurveSpec(8, "genmol_denovo_sgrpo_ng8_rewardsum_loo_2000"),
    CurveSpec(16, "genmol_denovo_sgrpo_ng16_rewardsum_loo_2000"),
    CurveSpec(32, "genmol_denovo_sgrpo_ng32_rewardsum_loo_2000"),
    CurveSpec(64, "genmol_denovo_sgrpo_rewardsum_loo_2000"),
)

DIVERSITY_WEIGHT_SPECS = (
    CurveSpec(0.0, "genmol_denovo_grpo_2000"),
    CurveSpec(0.1, "genmol_denovo_sgrpo_gw01_rewardsum_loo_2000"),
    CurveSpec(0.3, "genmol_denovo_sgrpo_gw03_rewardsum_loo_2000"),
    CurveSpec(0.5, "genmol_denovo_sgrpo_gw05_rewardsum_loo_2000"),
    CurveSpec(0.7, "genmol_denovo_sgrpo_gw07_rewardsum_loo_2000"),
    CurveSpec(0.9, "genmol_denovo_sgrpo_rewardsum_loo_2000"),
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.labelsize": 20,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
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
            "savefig.pad_inches": 0.04,
        }
    )


def strictly_dominates(left: Point, right: Point) -> bool:
    return (
        left.utility >= right.utility
        and left.diversity >= right.diversity
        and (left.utility > right.utility or left.diversity > right.diversity)
    )


def non_dominated(points: list[Point]) -> list[Point]:
    survivors = []
    for point in points:
        if not any(strictly_dominates(other, point) for other in points if other is not point):
            survivors.append(point)
    if not survivors:
        raise ValueError("Expected at least one non-dominated point")
    unique = sorted({(point.utility, point.diversity) for point in survivors})
    frontier = [Point(utility, diversity) for utility, diversity in unique]
    for index in range(len(frontier) - 1):
        if frontier[index + 1].diversity > frontier[index].diversity + 1e-12:
            raise ValueError("Expected monotone non-dominated frontier")
    return frontier


def compute_hv(points: list[Point], reference_point: Point) -> float:
    frontier = non_dominated(points)
    hv = 0.0
    previous_utility = reference_point.utility
    for point in frontier:
        if point.utility < reference_point.utility - 1e-12:
            raise ValueError("Point utility is worse than the reference point")
        if point.diversity < reference_point.diversity - 1e-12:
            raise ValueError("Point diversity is worse than the reference point")
        hv += (point.utility - previous_utility) * (point.diversity - reference_point.diversity)
        previous_utility = point.utility
    return hv


def load_de_novo_points() -> dict[str, list[Point]]:
    rows = load_json(DE_NOVO_RESULTS_PATH)
    if not isinstance(rows, list):
        raise TypeError(f"Expected list in {DE_NOVO_RESULTS_PATH}, got {type(rows).__name__}")
    points_by_experiment: dict[str, list[Point]] = {}
    for experiment in {row["experiment"] for row in rows}:
        subset = [row for row in rows if row["experiment"] == experiment]
        row_by_label = {str(row["sweep_label"]): row for row in subset}
        missing = [label for label in EXPECTED_PAIRED_SWEEP_LABELS if label not in row_by_label]
        if missing:
            continue
        points_by_experiment[experiment] = [
            Point(
                utility=float(row_by_label[label]["soft_reward_mean"]),
                diversity=float(row_by_label[label]["diversity"]),
            )
            for label in EXPECTED_PAIRED_SWEEP_LABELS
        ]
    return points_by_experiment


def compute_curve_hv(points_by_experiment: dict[str, list[Point]], curve_specs: tuple[CurveSpec, ...]) -> tuple[list[float], list[float], Point]:
    all_points = [point for spec in curve_specs for point in points_by_experiment[spec.experiment]]
    reference_point = Point(
        utility=min(point.utility for point in all_points),
        diversity=min(point.diversity for point in all_points),
    )
    x_values = [spec.x_value for spec in curve_specs]
    hv_values = [compute_hv(points_by_experiment[spec.experiment], reference_point) for spec in curve_specs]
    return x_values, hv_values, reference_point


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(False)
    ax.set_box_aspect(0.5)


def plot_curve(ax: plt.Axes, x_values: list[float], y_values: list[float], color: str) -> None:
    ax.plot(
        x_values,
        y_values,
        color=color,
        linewidth=2.8,
        marker="o",
        markersize=7.8,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=1.0,
        solid_capstyle="round",
        solid_joinstyle="round",
    )


def annotate_grpo(
    ax: plt.Axes,
    x_value: float,
    y_value: float,
    text_x: float,
    text_y: float,
) -> None:
    ax.annotate(
        "GRPO",
        xy=(x_value, y_value),
        xytext=(text_x, text_y),
        textcoords="data",
        ha="center",
        va="bottom",
        fontsize=18,
        color="#333333",
        arrowprops={
            "arrowstyle": "->",
            "color": "#555555",
            "lw": 1.5,
            "shrinkA": 2.0,
            "shrinkB": 4.0,
        },
        zorder=5,
    )


def plot() -> Path:
    configure_style()
    points_by_experiment = load_de_novo_points()

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 3.8))
    fig.patch.set_facecolor("white")

    group_size_x, group_size_hv, _ = compute_curve_hv(points_by_experiment, GROUP_SIZE_SPECS)
    diversity_weight_x, diversity_weight_hv, _ = compute_curve_hv(points_by_experiment, DIVERSITY_WEIGHT_SPECS)

    left_ax, right_ax = axes
    style_axis(left_ax)
    style_axis(right_ax)

    plot_curve(left_ax, group_size_x, group_size_hv, GROUP_SIZE_COLOR)
    plot_curve(right_ax, diversity_weight_x, diversity_weight_hv, DIVERSITY_WEIGHT_COLOR)

    left_ax.set_xlabel("Group Size")
    left_ax.set_ylabel("Hypervolume")
    left_ax.set_xticks(group_size_x)

    right_ax.set_xlabel("Weight of Diversity Rewards")
    right_ax.set_ylabel("Hypervolume")
    right_ax.set_xticks(diversity_weight_x)
    right_ax.set_xticklabels(["0", "0.1", "0.3", "0.5", "0.7", "0.9"])

    annotate_grpo(
        left_ax,
        x_value=group_size_x[0],
        y_value=group_size_hv[0],
        text_x=10.7,
        text_y=group_size_hv[0] + 0.0006,
    )
    annotate_grpo(
        right_ax,
        x_value=diversity_weight_x[0],
        y_value=diversity_weight_hv[0],
        text_x=0.13,
        text_y=diversity_weight_hv[0] + 0.0011,
    )

    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.22, top=0.98, wspace=0.20)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)
    return OUTPUT_PATH


if __name__ == "__main__":
    output_path = plot()
    print(output_path)
