#!/usr/bin/env python3
"""Compute main-figure Pareto metrics and emit a paper table."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VIS_CODE_DIR = REPO_ROOT / "vis-code"
if str(VIS_CODE_DIR) not in sys.path:
    sys.path.append(str(VIS_CODE_DIR))

from main_pareto_common import PANEL_SPECS, load_panel_series


OUTPUT_PATH = REPO_ROOT / "figs" / "main-table.md"
R2_NUM_WEIGHTS = 101
IDEAL_POINT = (1.0, 1.0)


@dataclass(frozen=True)
class Point:
    utility: float
    diversity: float


@dataclass(frozen=True)
class ModelMetrics:
    hv: float
    dist_to_ideal: float
    r2: float


@dataclass(frozen=True)
class GroupMetrics:
    title: str
    reference_point: Point
    per_model: dict[str, ModelMetrics]


def to_points(plot_points) -> list[Point]:
    points = [Point(float(point.x), float(point.y)) for point in plot_points]
    if not points:
        raise ValueError("Expected at least one point")
    return points


def strictly_dominates(left: Point, right: Point) -> bool:
    return (
        left.utility >= right.utility
        and left.diversity >= right.diversity
        and (left.utility > right.utility or left.diversity > right.diversity)
    )


def non_dominated(points: list[Point]) -> list[Point]:
    result = []
    for point in points:
        if not any(strictly_dominates(other, point) for other in points if other is not point):
            result.append(point)
    if not result:
        raise ValueError("Expected at least one non-dominated point")
    result = sorted({(point.utility, point.diversity) for point in result})
    frontier = [Point(utility, diversity) for utility, diversity in result]
    for index in range(len(frontier) - 1):
        if frontier[index + 1].diversity > frontier[index].diversity + 1e-12:
            raise ValueError("Non-dominated frontier must be monotone decreasing in diversity")
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


def compute_dist_to_ideal(points: list[Point]) -> float:
    ideal_utility, ideal_diversity = IDEAL_POINT
    return min(
        math.sqrt((ideal_utility - point.utility) ** 2 + (ideal_diversity - point.diversity) ** 2)
        for point in points
    )


def r2_weights(num_weights: int) -> list[tuple[float, float]]:
    if num_weights < 2:
        raise ValueError("R2_NUM_WEIGHTS must be at least 2")
    return [
        (index / float(num_weights - 1), 1.0 - index / float(num_weights - 1))
        for index in range(num_weights)
    ]


def compute_r2(points: list[Point], num_weights: int) -> float:
    weights = r2_weights(num_weights)
    losses = []
    for weight_utility, weight_diversity in weights:
        losses.append(
            min(
                max(
                    weight_utility * (1.0 - point.utility),
                    weight_diversity * (1.0 - point.diversity),
                )
                for point in points
            )
        )
    return sum(losses) / float(len(losses))


def compute_group_metrics() -> list[GroupMetrics]:
    groups: list[GroupMetrics] = []
    for panel_spec in PANEL_SPECS:
        series = load_panel_series(panel_spec)
        all_points = [point for points in series.values() for point in to_points(points)]
        reference_point = Point(
            utility=min(point.utility for point in all_points),
            diversity=min(point.diversity for point in all_points),
        )
        per_model = {}
        for model_spec in panel_spec.models:
            model_points = to_points(series[model_spec.legend_label])
            per_model[model_spec.legend_label] = ModelMetrics(
                hv=compute_hv(model_points, reference_point),
                dist_to_ideal=compute_dist_to_ideal(model_points),
                r2=compute_r2(model_points, R2_NUM_WEIGHTS),
            )
        groups.append(
            GroupMetrics(
                title=panel_spec.title,
                reference_point=reference_point,
                per_model=per_model,
            )
        )
    return groups


def format_value(value: float) -> str:
    return f"{value:.4f}"


def build_table(groups: list[GroupMetrics]) -> str:
    metric_rows = (
        ("HV ↑", "hv"),
        ("Dist. to Ideal ↓", "dist_to_ideal"),
        ("R2 ↓", "r2"),
    )
    lines = [
        "# Main Table",
        "",
        "<table>",
        "  <thead>",
        "    <tr>",
        '      <th rowspan="2">Metric</th>',
    ]
    for group in groups:
        lines.append(f'      <th colspan="{len(group.per_model)}">{group.title}</th>')
    lines.extend(
        [
            "    </tr>",
            "    <tr>",
        ]
    )
    for group in groups:
        for model_label in group.per_model:
            lines.append(f"      <th>{model_label}</th>")
    lines.extend(
        [
            "    </tr>",
            "  </thead>",
            "  <tbody>",
        ]
    )
    for metric_label, field_name in metric_rows:
        lines.append("    <tr>")
        lines.append(f"      <td>{metric_label}</td>")
        for group in groups:
            for model_label in group.per_model:
                metrics = group.per_model[model_label]
                lines.append(f"      <td>{format_value(getattr(metrics, field_name))}</td>")
        lines.append("    </tr>")
    lines.extend(
        [
            "  </tbody>",
            "</table>",
            "",
            "## Notes",
            "",
            f"- Assumption: `R2` uses `L = {R2_NUM_WEIGHTS}` uniformly spaced 2D weights from `(0, 1)` to `(1, 0)`.",
            "- `HV` is computed per group with the panel-level bad reference point `(min utility, min diversity)` taken over all points shown in that panel.",
            "- `Dist. to Ideal` uses the ideal point `(1, 1)` and takes the minimum Euclidean distance over the model's sweep points.",
            "",
            "## Group Reference Points",
            "",
        ]
    )
    for group in groups:
        lines.append(
            f"- `{group.title}`: `({group.reference_point.utility:.4f}, {group.reference_point.diversity:.4f})`"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    groups = compute_group_metrics()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_table(groups))
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
