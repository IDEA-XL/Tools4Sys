#!/usr/bin/env python3
"""Shared config helpers for GenMol de novo diversity-geometry dynamics figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DynamicsModelSpec:
    model_id: str
    row_id: str
    row_title: str
    slot_index: int
    slot_label: str
    checkpoint_path: str
    display_name: str


@dataclass(frozen=True)
class DynamicsConfig:
    output_dir: str
    figure_output_dir: str
    seed: int
    bf16: bool
    device: str
    num_samples: int
    generation_batch_size: int
    generation_temperature: float
    randomness: float
    min_add_len: int
    max_completion_length: int | None
    length_path: str | None
    reward_weights: dict
    soft_reward_marker_threshold: float
    reduction_tsne_perplexity: float
    reduction_tsne_n_iter: int
    reduction_umap_n_neighbors: int
    reduction_umap_min_dist: float
    models: list[DynamicsModelSpec]


def load_dynamics_config(path: str | Path) -> DynamicsConfig:
    config_path = Path(path)
    with config_path.open() as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict config in {config_path}, got {type(raw).__name__}")
    raw_models = raw.pop("models", None)
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("models must be a non-empty list")
    models = [DynamicsModelSpec(**item) for item in raw_models]
    config = DynamicsConfig(models=models, **raw)
    _validate_dynamics_config(config)
    return config


def _validate_dynamics_config(config: DynamicsConfig) -> None:
    if config.num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {config.num_samples}")
    if config.generation_batch_size <= 0:
        raise ValueError("generation_batch_size must be positive")
    if float(config.generation_temperature) <= 0.0:
        raise ValueError("generation_temperature must be positive")
    if float(config.randomness) <= 0.0:
        raise ValueError("randomness must be positive")
    if config.min_add_len <= 0:
        raise ValueError("min_add_len must be positive")
    if not 0.0 <= float(config.soft_reward_marker_threshold) <= 1.0:
        raise ValueError("soft_reward_marker_threshold must be in [0, 1]")
    if config.reduction_tsne_perplexity <= 0.0:
        raise ValueError("reduction_tsne_perplexity must be positive")
    if config.reduction_tsne_n_iter <= 0:
        raise ValueError("reduction_tsne_n_iter must be positive")
    if config.reduction_umap_n_neighbors <= 1:
        raise ValueError("reduction_umap_n_neighbors must be greater than 1")
    if not 0.0 <= float(config.reduction_umap_min_dist) <= 1.0:
        raise ValueError("reduction_umap_min_dist must be in [0, 1]")
    if not isinstance(config.reward_weights, dict):
        raise TypeError("reward_weights must be a dict")
    if not config.models:
        raise ValueError("models must be non-empty")
    model_ids = [model.model_id for model in config.models]
    if len(set(model_ids)) != len(model_ids):
        raise ValueError(f"Duplicate model_id values: {model_ids}")
    slot_indices = [int(model.slot_index) for model in config.models]
    if min(slot_indices) < 0:
        raise ValueError("slot_index must be non-negative")


def output_dir_path(config: DynamicsConfig) -> Path:
    return Path(config.output_dir)


def model_output_dir(config: DynamicsConfig, model: DynamicsModelSpec) -> Path:
    return output_dir_path(config) / "per_model" / model.model_id
