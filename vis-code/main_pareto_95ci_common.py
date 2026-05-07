#!/usr/bin/env python3
"""Shared helpers for 5-run mean/95% CI main-figure aggregation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from main_pareto_common import (
    PANEL_SPECS,
    PanelPoint,
    PanelSpec,
    build_denovo_series,
    build_mmgenmol_series,
    build_progen2_series,
    load_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RESULTS_ROOT = REPO_ROOT / "sgrpo-main-results" / "ci95"
RUN_SEEDS = (42, 43, 44, 45, 46)

_T_CRITICAL_95_BY_DF = {
    1: 12.7062047364,
    2: 4.30265272975,
    3: 3.18244630528,
    4: 2.7764451052,
    5: 2.57058183661,
    6: 2.44691185114,
    7: 2.36462425101,
    8: 2.30600413503,
    9: 2.26215716285,
    10: 2.22813885196,
    11: 2.20098516008,
    12: 2.17881282966,
    13: 2.16036865646,
    14: 2.14478668792,
    15: 2.13144954556,
    16: 2.11990529922,
    17: 2.10981557783,
    18: 2.10092204024,
    19: 2.09302405441,
    20: 2.08596344727,
    21: 2.07961384473,
    22: 2.0738730679,
    23: 2.06865761042,
    24: 2.06389856163,
    25: 2.05953855275,
    26: 2.05552943864,
    27: 2.05183051648,
    28: 2.0484071418,
    29: 2.04522964213,
    30: 2.0422724563,
}


@dataclass(frozen=True)
class AggregatePoint:
    x_mean: float
    x_ci95: float
    y_mean: float
    y_ci95: float
    sweep_rank: int
    model: object
    run_values_x: tuple[float, ...]
    run_values_y: tuple[float, ...]


def _default_run_paths_for_panel(spec: PanelSpec) -> tuple[Path, ...]:
    if spec.source_kind == "denovo_paired":
        leaf = "denovo"
    elif spec.source_kind == "mmgenmol_paired":
        leaf = "mmgenmol"
    elif spec.source_kind == "progen2_temperature":
        leaf = "progen2"
    else:
        raise ValueError(f"Unsupported source kind: {spec.source_kind}")
    return tuple(LOCAL_RESULTS_ROOT / leaf / f"seed{seed}.json" for seed in RUN_SEEDS)


def _t_critical_95(num_samples: int) -> float:
    if num_samples < 2:
        raise ValueError("At least two runs are required to compute a confidence interval")
    degrees_of_freedom = num_samples - 1
    if degrees_of_freedom in _T_CRITICAL_95_BY_DF:
        return _T_CRITICAL_95_BY_DF[degrees_of_freedom]
    return 1.959963984540054


def mean_ci95(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Expected at least one value")
    for value in values:
        if not math.isfinite(value):
            raise ValueError(f"Encountered non-finite value: {value}")
    mean = sum(values) / float(len(values))
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / float(len(values) - 1)
    standard_error = math.sqrt(variance) / math.sqrt(len(values))
    return mean, _t_critical_95(len(values)) * standard_error


def load_panel_series_from_results_path(spec: PanelSpec, results_path: Path) -> dict[str, list[PanelPoint]]:
    data = load_json(results_path)
    if spec.source_kind == "denovo_paired":
        if not isinstance(data, list):
            raise TypeError(f"Expected list in {results_path}, got {type(data).__name__}")
        return {model.legend_label: build_denovo_series(data, model) for model in spec.models}
    if spec.source_kind == "mmgenmol_paired":
        if not isinstance(data, list):
            raise TypeError(f"Expected list in {results_path}, got {type(data).__name__}")
        return {model.legend_label: build_mmgenmol_series(data, model) for model in spec.models}
    if spec.source_kind == "progen2_temperature":
        if not isinstance(data, dict) or "results" not in data:
            raise TypeError(f"Expected dict with results in {results_path}")
        results = data["results"]
        if not isinstance(results, list):
            raise TypeError(f"Expected results list in {results_path}")
        return {model.legend_label: build_progen2_series(results, model) for model in spec.models}
    raise ValueError(f"Unsupported source kind: {spec.source_kind}")


def load_panel_runs(
    spec: PanelSpec,
    run_paths: tuple[Path, ...] | None = None,
) -> list[dict[str, list[PanelPoint]]]:
    resolved_paths = run_paths or _default_run_paths_for_panel(spec)
    missing = [path for path in resolved_paths if not path.is_file()]
    if missing:
        missing_str = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing run summaries for {spec.title}:\n{missing_str}")
    return [load_panel_series_from_results_path(spec, path) for path in resolved_paths]


def aggregate_panel_runs(
    spec: PanelSpec,
    run_paths: tuple[Path, ...] | None = None,
) -> dict[str, list[AggregatePoint]]:
    panel_runs = load_panel_runs(spec, run_paths=run_paths)
    aggregated: dict[str, list[AggregatePoint]] = {}
    for model in spec.models:
        label = model.legend_label
        per_run_points = [panel_run[label] for panel_run in panel_runs]
        num_points = len(per_run_points[0])
        if num_points == 0:
            raise ValueError(f"Expected non-empty series for {spec.title} / {label}")
        for run_index, points in enumerate(per_run_points[1:], start=1):
            if len(points) != num_points:
                raise ValueError(
                    f"Inconsistent sweep length for {spec.title} / {label}: "
                    f"run0={num_points}, run{run_index}={len(points)}"
                )
        aggregate_points: list[AggregatePoint] = []
        for point_index in range(num_points):
            aligned_points = [points[point_index] for points in per_run_points]
            sweep_ranks = {int(point.sweep_rank) for point in aligned_points}
            if len(sweep_ranks) != 1:
                raise ValueError(
                    f"Mismatched sweep ranks for {spec.title} / {label} / point {point_index}: "
                    f"{sorted(sweep_ranks)}"
                )
            xs = [float(point.x) for point in aligned_points]
            ys = [float(point.y) for point in aligned_points]
            x_mean, x_ci95 = mean_ci95(xs)
            y_mean, y_ci95 = mean_ci95(ys)
            aggregate_points.append(
                AggregatePoint(
                    x_mean=x_mean,
                    x_ci95=x_ci95,
                    y_mean=y_mean,
                    y_ci95=y_ci95,
                    sweep_rank=int(aligned_points[0].sweep_rank),
                    model=aligned_points[0].model,
                    run_values_x=tuple(xs),
                    run_values_y=tuple(ys),
                )
            )
        aggregated[label] = aggregate_points
    return aggregated


__all__ = [
    "AggregatePoint",
    "LOCAL_RESULTS_ROOT",
    "PANEL_SPECS",
    "REPO_ROOT",
    "RUN_SEEDS",
    "aggregate_panel_runs",
    "load_panel_runs",
    "load_panel_series_from_results_path",
    "mean_ci95",
]
