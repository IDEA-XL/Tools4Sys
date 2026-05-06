#!/usr/bin/env python3
"""Plot the paper main-result Pareto figure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from main_pareto_common import (
    COLOR_GRPO,
    COLOR_MEMORY,
    COLOR_ORIGINAL,
    COLOR_SGRPO,
    ModelSpec,
    PANEL_SPECS,
    PanelPoint,
    load_panel_series,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "figs" / "main-pareto.pdf"
NON_PARETO_ALPHA = 0.25
SHADE_ALPHA = 0.10
UNIFORM_FONT_SIZE = 21
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
    y: float
    model: ModelSpec
    sweep_rank: int
    model_pareto: bool = False
    global_pareto: bool = False


@dataclass(frozen=True)
class PlotSegment:
    start: PlotPoint
    end: PlotPoint


def configure_style() -> None:
    font_family = "DejaVu Sans"
    if APTOS_DISPLAY_FONT_PATH.exists():
        font_manager.fontManager.addfont(str(APTOS_DISPLAY_FONT_PATH))
        font_family = "Aptos"
    print(f"Using font family: {font_family}")
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


def to_plot_points(panel_points: list[PanelPoint]) -> list[PlotPoint]:
    return [
        PlotPoint(
            x=float(point.x),
            y=float(point.y),
            model=point.model,
            sweep_rank=int(point.sweep_rank),
        )
        for point in panel_points
    ]


def strictly_dominated(point: PlotPoint, others: list[PlotPoint]) -> bool:
    return any(other.x > point.x and other.y > point.y for other in others if other is not point)


def mark_model_pareto(points_by_model: dict[str, list[PlotPoint]]) -> list[PlotPoint]:
    all_points = [point for points in points_by_model.values() for point in points]
    for points in points_by_model.values():
        for point in points:
            point.model_pareto = not strictly_dominated(point, points)
    return all_points


def sorted_model_frontier(points: list[PlotPoint]) -> list[PlotPoint]:
    frontier = [point for point in points if point.model_pareto]
    if not frontier:
        raise ValueError("Expected at least one model Pareto point")
    frontier.sort(key=lambda point: (point.x, -point.y))
    for idx in range(len(frontier) - 1):
        if frontier[idx + 1].y > frontier[idx].y + 1e-9:
            raise ValueError("Model Pareto frontier is not monotone after sorting by utility")
    return frontier


def build_model_pareto_polygon(frontier: list[PlotPoint], x_min: float, y_min: float) -> list[tuple[float, float]]:
    polygon = [(x_min, y_min), (x_min, frontier[0].y)]
    polygon.extend((point.x, point.y) for point in frontier)
    polygon.append((frontier[-1].x, y_min))
    return polygon


def build_segments(
    points_by_model: dict[str, list[PlotPoint]],
) -> tuple[list[PlotSegment], list[PlotSegment], dict[str, list[PlotPoint]]]:
    sweep_segments: list[PlotSegment] = []
    frontier_segments: list[PlotSegment] = []
    frontier_points_by_model: dict[str, list[PlotPoint]] = {}
    for label, points in points_by_model.items():
        ordered_points = sorted(points, key=lambda point: point.sweep_rank)
        for left, right in zip(ordered_points, ordered_points[1:]):
            sweep_segments.append(PlotSegment(left, right))
        frontier_points = sorted_model_frontier(ordered_points)
        frontier_points_by_model[label] = frontier_points
        for left, right in zip(frontier_points, frontier_points[1:]):
            frontier_segments.append(PlotSegment(left, right))
    return sweep_segments, frontier_segments, frontier_points_by_model


def segment_has_upper_right_point(segment: PlotSegment, point: PlotPoint) -> bool:
    epsilon = 1e-12
    low = 0.0
    high = 1.0
    for start, delta, threshold in (
        (segment.start.x, segment.end.x - segment.start.x, point.x),
        (segment.start.y, segment.end.y - segment.start.y, point.y),
    ):
        if abs(delta) <= epsilon:
            if start <= threshold + epsilon:
                return False
            continue
        boundary = (threshold - start) / delta
        if delta > 0.0:
            low = max(low, boundary)
        else:
            high = min(high, boundary)
        if high - low <= epsilon:
            return False
    return high - low > epsilon


def mark_global_pareto(all_points: list[PlotPoint], all_segments: list[PlotSegment]) -> None:
    for point in all_points:
        point.global_pareto = (
            not strictly_dominated(point, all_points)
            and not any(segment_has_upper_right_point(segment, point) for segment in all_segments)
        )


def axis_limits(points: list[PlotPoint]) -> tuple[float, float, float, float]:
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    x_pad = max(x_span * 0.08, 0.015)
    y_pad = max(y_span * 0.12, 0.02)
    return min(xs) - x_pad, max(xs) + x_pad * 0.65, max(0.0, min(ys) - y_pad), min(1.0, max(ys) + y_pad * 0.45)


def draw_panel(ax: plt.Axes, panel_spec) -> None:
    raw_series = load_panel_series(panel_spec)
    points_by_model = {label: to_plot_points(points) for label, points in raw_series.items()}
    all_points = mark_model_pareto(points_by_model)
    sweep_segments, frontier_segments, frontier_points_by_model = build_segments(points_by_model)
    mark_global_pareto(all_points, sweep_segments + frontier_segments)
    x_min, x_max, y_min, y_max = axis_limits(all_points)

    ax.set_facecolor("white")
    ax.grid(False)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(panel_spec.title, pad=10)

    for model in panel_spec.models:
        points = points_by_model[model.legend_label]
        points.sort(key=lambda point: point.sweep_rank)
        frontier_points = frontier_points_by_model[model.legend_label]

        ax.add_patch(
            Polygon(
                build_model_pareto_polygon(frontier_points, x_min, y_min),
                closed=True,
                facecolor=model.color,
                edgecolor="none",
                alpha=SHADE_ALPHA,
                zorder=0,
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
                zorder=2,
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
            marker = "*" if point.global_pareto else model.marker
            size = 240 if point.global_pareto else 92
            alpha = 1.0 if point.model_pareto else NON_PARETO_ALPHA
            edge_width = 1.3 if point.global_pareto else 1.0
            ax.scatter(
                [point.x],
                [point.y],
                s=size,
                marker=marker,
                facecolor=model.color,
                edgecolor="white",
                linewidths=edge_width,
                alpha=alpha,
                zorder=4 if point.global_pareto else 3,
            )


def build_legend_handles() -> list[Line2D]:
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
            color="none",
            marker="*",
            markersize=14,
            markerfacecolor=COLOR_SGRPO,
            markeredgecolor="white",
            markeredgewidth=1.1,
            label="Overall Non-Dominated",
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
    fig, axes = plt.subplots(1, 3, figsize=(18.2, 5.8))
    fig.patch.set_facecolor("white")

    for ax, panel_spec in zip(axes, PANEL_SPECS):
        draw_panel(ax, panel_spec)

    fig.supxlabel("Utility", y=0.095, fontsize=UNIFORM_FONT_SIZE)
    fig.supylabel("Diversity", x=0.03, fontsize=UNIFORM_FONT_SIZE)
    fig.legend(
        handles=build_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=6,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.25,
        handletextpad=0.55,
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.88, bottom=0.22, wspace=0.18)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)
    return OUTPUT_PATH


if __name__ == "__main__":
    output_path = plot()
    print(output_path)
