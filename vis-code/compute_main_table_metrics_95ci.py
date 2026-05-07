#!/usr/bin/env python3
"""Compute 5-run mean/95% CI main-figure Pareto metrics and emit a paper table."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from main_pareto_95ci_common import REPO_ROOT, PANEL_SPECS, load_panel_runs, mean_ci95


OUTPUT_PATH = REPO_ROOT / "figs" / "main-table-95ci.md"
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
class MetricSummary:
    mean: float
    ci95: float


def _to_points(plot_points) -> list[Point]:
    points = [Point(float(point.x), float(point.y)) for point in plot_points]
    if not points:
        raise ValueError("Expected at least one point")
    return points


def _strictly_dominates(left: Point, right: Point) -> bool:
    return (
        left.utility >= right.utility
        and left.diversity >= right.diversity
        and (left.utility > right.utility or left.diversity > right.diversity)
    )


def _non_dominated(points: list[Point]) -> list[Point]:
    result = []
    for point in points:
        if not any(_strictly_dominates(other, point) for other in points if other is not point):
            result.append(point)
    if not result:
        raise ValueError("Expected at least one non-dominated point")
    deduped = sorted({(point.utility, point.diversity) for point in result})
    frontier = [Point(utility, diversity) for utility, diversity in deduped]
    for index in range(len(frontier) - 1):
        if frontier[index + 1].diversity > frontier[index].diversity + 1e-12:
            raise ValueError("Non-dominated frontier must be monotone decreasing in diversity")
    return frontier


def _compute_hv(points: list[Point], reference_point: Point) -> float:
    frontier = _non_dominated(points)
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


def _compute_dist_to_ideal(points: list[Point]) -> float:
    ideal_utility, ideal_diversity = IDEAL_POINT
    return min(
        math.sqrt((ideal_utility - point.utility) ** 2 + (ideal_diversity - point.diversity) ** 2)
        for point in points
    )


def _r2_weights(num_weights: int) -> list[tuple[float, float]]:
    if num_weights < 2:
        raise ValueError("R2_NUM_WEIGHTS must be at least 2")
    return [
        (index / float(num_weights - 1), 1.0 - index / float(num_weights - 1))
        for index in range(num_weights)
    ]


def _compute_r2(points: list[Point], num_weights: int) -> float:
    weights = _r2_weights(num_weights)
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


def _format_summary(summary: MetricSummary) -> str:
    return f"{summary.mean:.4f} ± {summary.ci95:.4f}"


def _summarize(values: list[float]) -> MetricSummary:
    mean, ci95 = mean_ci95([float(value) for value in values])
    return MetricSummary(mean=mean, ci95=ci95)


def build_table() -> str:
    metric_rows = (
        ("HV ↑", "hv"),
        ("Dist. to Ideal ↓", "dist_to_ideal"),
        ("R2 ↓", "r2"),
    )

    groups = []
    for panel_spec in PANEL_SPECS:
        panel_runs = load_panel_runs(panel_spec)
        per_run_metrics: list[dict[str, ModelMetrics]] = []
        per_run_reference_points: list[Point] = []
        for panel_run in panel_runs:
            all_points = [
                point
                for label in panel_run
                for point in _to_points(panel_run[label])
            ]
            reference_point = Point(
                utility=min(point.utility for point in all_points),
                diversity=min(point.diversity for point in all_points),
            )
            per_run_reference_points.append(reference_point)
            run_metrics: dict[str, ModelMetrics] = {}
            for model_spec in panel_spec.models:
                model_points = _to_points(panel_run[model_spec.legend_label])
                run_metrics[model_spec.legend_label] = ModelMetrics(
                    hv=_compute_hv(model_points, reference_point),
                    dist_to_ideal=_compute_dist_to_ideal(model_points),
                    r2=_compute_r2(model_points, R2_NUM_WEIGHTS),
                )
            per_run_metrics.append(run_metrics)

        summarized = {}
        for model_spec in panel_spec.models:
            label = model_spec.legend_label
            summarized[label] = {
                "hv": _summarize([run_metrics[label].hv for run_metrics in per_run_metrics]),
                "dist_to_ideal": _summarize(
                    [run_metrics[label].dist_to_ideal for run_metrics in per_run_metrics]
                ),
                "r2": _summarize([run_metrics[label].r2 for run_metrics in per_run_metrics]),
            }
        groups.append(
            {
                "title": panel_spec.title,
                "models": [model.legend_label for model in panel_spec.models],
                "summaries": summarized,
                "reference_points": per_run_reference_points,
            }
        )

    lines = [
        "# Main Table (5 Runs, Mean ± 95% CI)",
        "",
        "<table>",
        "  <thead>",
        "    <tr>",
        '      <th rowspan="2">Metric</th>',
    ]
    for group in groups:
        lines.append(f'      <th colspan="{len(group["models"])}">{group["title"]}</th>')
    lines.extend(
        [
            "    </tr>",
            "    <tr>",
        ]
    )
    for group in groups:
        for label in group["models"]:
            lines.append(f"      <th>{label}</th>")
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
            for label in group["models"]:
                lines.append(f"      <td>{_format_summary(group['summaries'][label][field_name])}</td>")
        lines.append("    </tr>")
    lines.extend(
        [
            "  </tbody>",
            "</table>",
            "",
            "## Notes",
            "",
            "- Each cell reports the mean and 95% confidence interval over 5 independent sweep runs.",
            "- Assumption: the 95% confidence interval uses the Student-t critical value for `n = 5` runs.",
            "- `HV` is computed separately inside each run using that run's panel-level bad reference point `(min utility, min diversity)` over all points shown in the corresponding panel.",
            f"- `R2` uses `L = {R2_NUM_WEIGHTS}` uniformly spaced 2D weights from `(0, 1)` to `(1, 0)`.",
            "- `Dist. to Ideal` uses the ideal point `(1, 1)` and takes the minimum Euclidean distance over the model's sweep points within each run.",
            "",
            "## Run-Level Reference Points",
            "",
        ]
    )
    for group in groups:
        refs = ", ".join(f"({ref.utility:.4f}, {ref.diversity:.4f})" for ref in group["reference_points"])
        lines.append(f"- `{group['title']}`: {refs}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_table())
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
