#!/usr/bin/env python3
"""Shared data definitions for the main Pareto figure and derived tables."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

DE_NOVO_RESULTS_PATH = (
    REPO_ROOT
    / "sgrpo-main-results"
    / "genmol-denovo"
    / "denovo_main_results_paired_sweep_20260504.json"
)
MMGENMOL_RESULTS_PATH = (
    REPO_ROOT
    / "sgrpo-main-results"
    / "mmgenmol"
    / "mmgenmol_paired_main_results_20260504.json"
)
PROGEN2_RESULTS_PATH = (
    REPO_ROOT / "sgrpo-main-results" / "progen2" / "progen2_temperature_sweep_20260503.json"
)

EXPECTED_PAIRED_SWEEP_LABELS = (
    "r=0.1,t=0.5",
    "r=0.3,t=0.8",
    "r=0.5,t=1.1",
    "r=0.7,t=1.4",
    "r=0.9,t=1.7",
    "r=1.0,t=2.0",
)
PAIRED_SWEEP_ORDER_HIGH_TO_LOW = tuple(reversed(EXPECTED_PAIRED_SWEEP_LABELS))
PROGEN2_TEMPERATURE_ORDER_HIGH_TO_LOW = (
    1.2,
    1.1,
    1.0,
    0.9,
    0.8,
    0.7,
    0.6,
    0.5,
    0.4,
    0.3,
    0.2,
    0.1,
)

COLOR_ORIGINAL = "#7A9E3A"
COLOR_GRPO = "#2A9D8F"
COLOR_SGRPO = "#1F4E79"
COLOR_MEMORY = "#B85C38"


@dataclass(frozen=True)
class ModelSpec:
    source_id: str
    legend_label: str
    color: str
    marker: str


@dataclass(frozen=True)
class PanelSpec:
    title: str
    source_kind: str
    results_path: Path
    models: tuple[ModelSpec, ...]


@dataclass(frozen=True)
class PanelPoint:
    x: float
    y: float
    model: ModelSpec
    sweep_rank: int


PANEL_SPECS = (
    PanelSpec(
        title="De Novo Molecule Design",
        source_kind="denovo_paired",
        results_path=DE_NOVO_RESULTS_PATH,
        models=(
            ModelSpec("original_genmol_v2", "Original", COLOR_ORIGINAL, "D"),
            ModelSpec("genmol_denovo_grpo_2000", "GRPO", COLOR_GRPO, "^"),
            ModelSpec("genmol_denovo_sgrpo_rewardsum_loo_2000", "SGRPO", COLOR_SGRPO, "o"),
            ModelSpec("genmol_denovo_grpo_hbd_2000", "Memory-Assisted GRPO", COLOR_MEMORY, "s"),
        ),
    ),
    PanelSpec(
        title="Pocket-Based Design",
        source_kind="mmgenmol_paired",
        results_path=MMGENMOL_RESULTS_PATH,
        models=(
            ModelSpec("original_5500", "Original", COLOR_ORIGINAL, "D"),
            ModelSpec("grpo_unidock_1000", "GRPO", COLOR_GRPO, "^"),
            ModelSpec("sgrpo_unidock_rewardsum_loo_1000", "SGRPO", COLOR_SGRPO, "o"),
        ),
    ),
    PanelSpec(
        title="De Novo Protein Design",
        source_kind="progen2_temperature",
        results_path=PROGEN2_RESULTS_PATH,
        models=(
            ModelSpec("original", "Original", COLOR_ORIGINAL, "D"),
            ModelSpec("grpo_step100", "GRPO", COLOR_GRPO, "^"),
            ModelSpec("sgrpo_gw08_rewardsum_loo_step100", "SGRPO", COLOR_SGRPO, "o"),
            ModelSpec("grpo_hbd_step100", "Memory-Assisted GRPO", COLOR_MEMORY, "s"),
        ),
    ),
)


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def validate_finite(value: float, name: str, row: dict[str, Any]) -> float:
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {name} value in row: {row}")
    return value


def build_denovo_series(rows: list[dict[str, Any]], model: ModelSpec) -> list[PanelPoint]:
    subset = [row for row in rows if row["experiment"] == model.source_id]
    if not subset:
        raise ValueError(f"Missing experiment {model.source_id} in {DE_NOVO_RESULTS_PATH}")
    row_by_label = {str(row["sweep_label"]): row for row in subset}
    missing = [label for label in EXPECTED_PAIRED_SWEEP_LABELS if label not in row_by_label]
    if missing:
        raise ValueError(f"Missing sweep labels for {model.source_id}: {missing}")
    points: list[PanelPoint] = []
    for rank, label in enumerate(PAIRED_SWEEP_ORDER_HIGH_TO_LOW):
        row = row_by_label[label]
        points.append(
            PanelPoint(
                x=validate_finite(float(row["soft_reward_mean"]), "soft_reward_mean", row),
                y=validate_finite(float(row["diversity"]), "diversity", row),
                model=model,
                sweep_rank=rank,
            )
        )
    return points


def build_mmgenmol_series(rows: list[dict[str, Any]], model: ModelSpec) -> list[PanelPoint]:
    subset = [row for row in rows if row["model_name"] == model.source_id]
    if not subset:
        raise ValueError(f"Missing model {model.source_id} in {MMGENMOL_RESULTS_PATH}")
    row_by_label: dict[str, dict[str, Any]] = {}
    for row in subset:
        label = f"r={row['randomness']:.1f},t={row['temperature']:.1f}"
        row_by_label[label] = row
    missing = [label for label in EXPECTED_PAIRED_SWEEP_LABELS if label not in row_by_label]
    if missing:
        raise ValueError(f"Missing paired sweep labels for {model.source_id}: {missing}")
    points: list[PanelPoint] = []
    for rank, label in enumerate(PAIRED_SWEEP_ORDER_HIGH_TO_LOW):
        row = row_by_label[label]
        points.append(
            PanelPoint(
                x=validate_finite(float(row["soft_reward_mean"]), "soft_reward_mean", row),
                y=validate_finite(float(row["diversity"]), "diversity", row),
                model=model,
                sweep_rank=rank,
            )
        )
    return points


def build_progen2_series(rows: list[dict[str, Any]], model: ModelSpec) -> list[PanelPoint]:
    subset = [row for row in rows if row["experiment"] == model.source_id]
    if not subset:
        raise ValueError(f"Missing experiment {model.source_id} in {PROGEN2_RESULTS_PATH}")
    row_by_temperature = {float(row["temperature"]): row for row in subset}
    missing = [temp for temp in PROGEN2_TEMPERATURE_ORDER_HIGH_TO_LOW if temp not in row_by_temperature]
    if missing:
        raise ValueError(f"Missing temperatures for {model.source_id}: {missing}")
    points: list[PanelPoint] = []
    for rank, temperature in enumerate(PROGEN2_TEMPERATURE_ORDER_HIGH_TO_LOW):
        row = row_by_temperature[temperature]
        points.append(
            PanelPoint(
                x=validate_finite(float(row["soft_reward_mean"]), "soft_reward_mean", row),
                y=validate_finite(float(row["diversity"]), "diversity", row),
                model=model,
                sweep_rank=rank,
            )
        )
    return points


def load_panel_series(spec: PanelSpec) -> dict[str, list[PanelPoint]]:
    data = load_json(spec.results_path)
    if spec.source_kind == "denovo_paired":
        if not isinstance(data, list):
            raise TypeError(f"Expected list in {spec.results_path}, got {type(data).__name__}")
        return {model.legend_label: build_denovo_series(data, model) for model in spec.models}
    if spec.source_kind == "mmgenmol_paired":
        if not isinstance(data, list):
            raise TypeError(f"Expected list in {spec.results_path}, got {type(data).__name__}")
        return {model.legend_label: build_mmgenmol_series(data, model) for model in spec.models}
    if spec.source_kind == "progen2_temperature":
        if not isinstance(data, dict) or "results" not in data:
            raise TypeError(f"Expected dict with results in {spec.results_path}")
        results = data["results"]
        if not isinstance(results, list):
            raise TypeError(f"Expected results list in {spec.results_path}")
        return {model.legend_label: build_progen2_series(results, model) for model in spec.models}
    raise ValueError(f"Unsupported source kind: {spec.source_kind}")
