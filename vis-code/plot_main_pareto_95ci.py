#!/usr/bin/env python3
"""Plot the paper main-result Pareto figure with 5-run mean and 95% CI error bars."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from main_pareto_common import ModelSpec
from main_pareto_95ci_common import PANEL_SPECS, REPO_ROOT, aggregate_panel_runs


OUTPUT_PATH = REPO_ROOT / "figs" / "main-pareto-95ci.pdf"
NON_PARETO_ALPHA = 0.25
SHADE_ALPHA = 0.10
UNIFORM_FONT_SIZE = 26
REFERENCE_LINE_COLOR = "#B5B5B5"
REFERENCE_MARKER_COLOR = "#6A6A6A"
REFERENCE_LINESTYLE = (0, (1.2, 2.6))
REFERENCE_LINEWIDTH = 1.15
ERRORBAR_LINEWIDTH = 1.55
APTOS_DISPLAY_FONT_PATH = (
    Path.home()
    / "Library"
    / "Group Containers"
    / "UBF8T346G9.Office"
    / "FontCache"
    / "4"
    / "CloudFonts"
    / "Aptos Display"
    / "32677218994.ttf"
)


@dataclass
class PlotPoint:
    x: float
    x_ci95: float
    y: float
    y_ci95: float
    model: ModelSpec
    sweep_rank: int
    model_pareto: bool = False


@dataclass(frozen=True)
class ReferencePoint:
    x: float
    y: float
    min_utility_point: PlotPoint
    min_diversity_point: PlotPoint


def configure_style() -> None:
    font_family = "DejaVu Sans"
    if APTOS_DISPLAY_FONT_PATH.exists():
        font_manager.fontManager.addfont(str(APTOS_DISPLAY_FONT_PATH))
        font_family = font_manager.FontProperties(fname=str(APTOS_DISPLAY_FONT_PATH)).get_name()
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": UNIFORM_FONT_SIZE,
            "axes.labelsize": UNIFORM_FONT_SIZE,
            "axes.titlesize": UNIFORM_FONT_SIZE,
            "xtick.labelsize": UNIFORM_FONT_SIZE,
            "ytick.labelsize": UNIFORM_FONT_SIZE,
            "legend.fontsize": UNIFORM_FONT_SIZE,
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


def _strictly_dominated(point: PlotPoint, others: list[PlotPoint]) -> bool:
    return any(other.x > point.x and other.y > point.y for other in others if other is not point)


def _mark_model_pareto(points_by_model: dict[str, list[PlotPoint]]) -> list[PlotPoint]:
    all_points = [point for points in points_by_model.values() for point in points]
    for points in points_by_model.values():
        for point in points:
            point.model_pareto = not _strictly_dominated(point, points)
    return all_points


def _sorted_model_frontier(points: list[PlotPoint]) -> list[PlotPoint]:
    frontier = [point for point in points if point.model_pareto]
    if not frontier:
        raise ValueError("Expected at least one model Pareto point")
    frontier.sort(key=lambda point: (point.x, -point.y))
    for idx in range(len(frontier) - 1):
        if frontier[idx + 1].y > frontier[idx].y + 1e-9:
            raise ValueError("Model Pareto frontier is not monotone after sorting by utility")
    return frontier


def _build_model_pareto_polygon(
    frontier: list[PlotPoint],
    x_min: float,
    y_min: float,
) -> list[tuple[float, float]]:
    polygon = [(x_min, y_min), (x_min, frontier[0].y)]
    polygon.extend((point.x, point.y) for point in frontier)
    polygon.append((frontier[-1].x, y_min))
    return polygon


def _compute_reference_point(all_points: list[PlotPoint]) -> ReferencePoint:
    reference_x = min(point.x for point in all_points)
    reference_y = min(point.y for point in all_points)
    min_utility_point = min(all_points, key=lambda point: (point.x, point.y))
    min_diversity_point = min(all_points, key=lambda point: (point.y, point.x))
    return ReferencePoint(
        x=reference_x,
        y=reference_y,
        min_utility_point=min_utility_point,
        min_diversity_point=min_diversity_point,
    )


def _axis_limits(points: list[PlotPoint]) -> tuple[float, float, float, float]:
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    x_pad = max(x_span * 0.08, 0.015)
    y_pad = max(y_span * 0.12, 0.02)
    return min(xs) - x_pad, max(xs) + x_pad * 0.65, max(0.0, min(ys) - y_pad), min(1.0, max(ys) + y_pad * 0.45)


def _set_panel_ticks(ax: plt.Axes, panel_spec, x_min: float, x_max: float) -> None:
    step = 0.1 if panel_spec.source_kind == "progen2_temperature" else 0.05
    start = math.ceil((x_min - 1e-12) / step) * step
    end = math.floor((x_max + 1e-12) / step) * step
    ticks: list[float] = []
    current = start
    while current <= end + 1e-12:
        ticks.append(round(current, 2))
        current += step
    if ticks:
        ax.set_xticks(ticks)


def _to_plot_points(panel_spec) -> dict[str, list[PlotPoint]]:
    aggregated = aggregate_panel_runs(panel_spec)
    points_by_model: dict[str, list[PlotPoint]] = {}
    for model in panel_spec.models:
        points_by_model[model.legend_label] = [
            PlotPoint(
                x=point.x_mean,
                x_ci95=point.x_ci95,
                y=point.y_mean,
                y_ci95=point.y_ci95,
                model=model,
                sweep_rank=point.sweep_rank,
            )
            for point in aggregated[model.legend_label]
        ]
    return points_by_model


def _draw_panel(ax: plt.Axes, panel_spec) -> None:
    points_by_model = _to_plot_points(panel_spec)
    all_points = _mark_model_pareto(points_by_model)
    frontier_points_by_model = {
        label: _sorted_model_frontier(points) for label, points in points_by_model.items()
    }
    reference_point = _compute_reference_point(all_points)
    x_min, x_max, y_min, y_max = _axis_limits(all_points)

    ax.set_facecolor("white")
    ax.grid(False)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    _set_panel_ticks(ax, panel_spec, x_min, x_max)
    ax.set_title(panel_spec.title, pad=12)
    ax.plot(
        [reference_point.x, reference_point.min_diversity_point.x],
        [reference_point.y, reference_point.y],
        color=REFERENCE_LINE_COLOR,
        linewidth=REFERENCE_LINEWIDTH,
        linestyle=REFERENCE_LINESTYLE,
        zorder=1.0,
    )
    ax.plot(
        [reference_point.x, reference_point.x],
        [reference_point.y, reference_point.min_utility_point.y],
        color=REFERENCE_LINE_COLOR,
        linewidth=REFERENCE_LINEWIDTH,
        linestyle=REFERENCE_LINESTYLE,
        zorder=1.0,
    )
    ax.scatter(
        [reference_point.x],
        [reference_point.y],
        s=120,
        marker="x",
        color=REFERENCE_MARKER_COLOR,
        linewidths=2.0,
        zorder=1.1,
    )

    for model in panel_spec.models:
        points = points_by_model[model.legend_label]
        points.sort(key=lambda point: point.sweep_rank)
        frontier_points = frontier_points_by_model[model.legend_label]

        ax.add_patch(
            Polygon(
                _build_model_pareto_polygon(frontier_points, x_min, y_min),
                closed=True,
                facecolor=model.color,
                edgecolor="none",
                alpha=SHADE_ALPHA,
                zorder=0.0,
            )
        )

        for left, right in zip(points, points[1:]):
            alpha = 1.0 if left.model_pareto and right.model_pareto else NON_PARETO_ALPHA
            ax.plot(
                [left.x, right.x],
                [left.y, right.y],
                color=model.color,
                linewidth=2.7,
                alpha=alpha,
                solid_capstyle="round",
                zorder=2.0,
            )

        for left, right in zip(frontier_points, frontier_points[1:]):
            ax.plot(
                [left.x, right.x],
                [left.y, right.y],
                color="white",
                linewidth=4.4,
                alpha=1.0,
                solid_capstyle="round",
                zorder=2.2,
            )
            ax.plot(
                [left.x, right.x],
                [left.y, right.y],
                color=model.color,
                linewidth=2.4,
                alpha=1.0,
                linestyle=(0, (4.5, 3.0)),
                solid_capstyle="round",
                zorder=2.3,
            )

        for point in points:
            alpha = 1.0 if point.model_pareto else NON_PARETO_ALPHA
            ax.errorbar(
                point.x,
                point.y,
                xerr=point.x_ci95,
                yerr=point.y_ci95,
                fmt="none",
                ecolor=model.color,
                elinewidth=ERRORBAR_LINEWIDTH,
                alpha=alpha,
                capsize=0.0,
                zorder=2.7,
            )
            ax.scatter(
                [point.x],
                [point.y],
                s=92,
                marker=model.marker,
                facecolor=model.color,
                edgecolor="white",
                linewidths=1.0,
                alpha=alpha,
                zorder=3.0,
            )


def _build_legend_handles() -> list[Line2D]:
    unique_models: dict[str, ModelSpec] = {}
    for panel in PANEL_SPECS:
        for model in panel.models:
            unique_models.setdefault(model.legend_label, model)
    ordered_labels = ("Original", "GRPO", "SGRPO", "Memory-Assisted GRPO")
    handles: list[Line2D] = []
    for label in ordered_labels:
        if label not in unique_models:
            continue
        model = unique_models[label]
        handles.append(
            Line2D(
                [0],
                [0],
                color=model.color,
                linewidth=2.8,
                marker=model.marker,
                markersize=9.5,
                markerfacecolor=model.color,
                markeredgecolor="white",
                markeredgewidth=1.1,
                label=model.legend_label,
            )
        )
    handles.append(
        Line2D(
            [0],
            [0],
            color=REFERENCE_MARKER_COLOR,
            marker="x",
            markersize=11,
            linestyle="None",
            markeredgewidth=2.0,
            label="Ref. Point of HV",
        )
    )
    handles.append(
        Line2D(
            [0],
            [0],
            color="#4A4A4A",
            linewidth=2.4,
            linestyle=(0, (4.5, 3.0)),
            label="Frontier",
        )
    )
    return handles


def plot() -> Path:
    configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(18.6, 6.1))
    fig.patch.set_facecolor("white")

    for ax, panel_spec in zip(axes, PANEL_SPECS):
        _draw_panel(ax, panel_spec)

    fig.supxlabel("Utility", y=0.088, fontsize=UNIFORM_FONT_SIZE)
    fig.supylabel("Diversity", x=0.018, fontsize=UNIFORM_FONT_SIZE)
    fig.legend(
        handles=_build_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=6,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.25,
        handletextpad=0.55,
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.88, bottom=0.23, wspace=0.18)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)
    return OUTPUT_PATH


if __name__ == "__main__":
    print(plot())
