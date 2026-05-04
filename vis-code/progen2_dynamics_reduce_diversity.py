#!/usr/bin/env python3
"""Reduce progen2 dynamics samples using training-time edit-sim diversity geometry."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))
sys.path.append(str(REPO_ROOT / "vis-code"))

from progen2_dynamics_common import load_dynamics_config, output_dir_path
from rl_shared.hbd import _get_hbd_cpu_worker_count, _get_sequence_distance_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--block-size", type=int, default=256)
    return parser.parse_args()


def _load_model_records(records_path: Path) -> list[dict]:
    rows = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No rows found in {records_path}")
    return rows


def _load_valid_rows(config, base_dir: Path) -> list[dict]:
    all_rows: list[dict] = []
    for model in config.models:
        model_dir = base_dir / "per_model" / model.model_id
        records_path = model_dir / "records.jsonl"
        if not records_path.is_file():
            raise FileNotFoundError(f"Missing records file: {records_path}")
        rows = _load_model_records(records_path)
        valid_rows = [row for row in rows if bool(row["is_valid"])]
        if not valid_rows:
            raise ValueError(f"No valid rows found for {model.model_id} in {records_path}")
        all_rows.extend(valid_rows)
    if not all_rows:
        raise ValueError("No valid rows found across all models")
    return all_rows


def _extract_sequences(rows: list[dict]) -> list[str]:
    sequences: list[str] = []
    for row in rows:
        sequence = str(row["sequence"]).strip().upper()
        if not sequence:
            raise ValueError(
                f"Encountered empty valid sequence for model_id={row['model_id']} sample_index={row['sample_index']}"
            )
        sequences.append(sequence)
    return sequences


def _compute_distance_matrix(sequences: list[str], *, block_size: int) -> np.ndarray:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    np_backend, rapidfuzz_process, Levenshtein = _get_sequence_distance_backend()
    worker_count = _get_hbd_cpu_worker_count()
    count = len(sequences)
    distance_matrix = np_backend.empty((count, count), dtype=np_backend.float32)
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        similarities = rapidfuzz_process.cdist(
            sequences[start:stop],
            sequences,
            scorer=Levenshtein.normalized_similarity,
            workers=worker_count,
            dtype=np_backend.float32,
        )
        if similarities.shape != (stop - start, count):
            raise RuntimeError(
                "Unexpected similarity block shape: "
                f"expected {(stop - start, count)} got {similarities.shape}"
            )
        distance_matrix[start:stop, :] = 1.0 - similarities
    np_backend.fill_diagonal(distance_matrix, 0.0)
    max_asymmetry = float(np_backend.max(np_backend.abs(distance_matrix - distance_matrix.T)))
    if max_asymmetry > 1e-6:
        raise RuntimeError(f"Edit-distance matrix is not symmetric enough: max_asymmetry={max_asymmetry}")
    min_value = float(distance_matrix.min())
    max_value = float(distance_matrix.max())
    if min_value < -1e-6 or max_value > 1.0 + 1e-6:
        raise RuntimeError(f"Distance matrix values out of range: min={min_value} max={max_value}")
    return distance_matrix


def main() -> None:
    args = parse_args()
    config = load_dynamics_config(args.config)
    base_dir = Path(args.output_dir) if args.output_dir else output_dir_path(config)
    base_dir.mkdir(parents=True, exist_ok=True)

    all_rows = _load_valid_rows(config, base_dir)
    sequences = _extract_sequences(all_rows)
    distance_matrix = _compute_distance_matrix(sequences, block_size=int(args.block_size))

    try:
        import umap
    except Exception as exc:
        raise ImportError("umap-learn is required for progen2 diversity dynamics UMAP reduction") from exc

    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=int(config.reduction_umap_n_neighbors),
        min_dist=float(config.reduction_umap_min_dist),
        metric="precomputed",
        random_state=int(config.seed),
        low_memory=True,
    )
    umap_coords = umap_model.fit_transform(distance_matrix)

    tsne_kwargs = {
        "n_components": 2,
        "perplexity": float(config.reduction_tsne_perplexity),
        "init": "random",
        "learning_rate": "auto",
        "metric": "precomputed",
        "random_state": int(config.seed),
        "verbose": 1,
    }
    tsne_signature = inspect.signature(TSNE)
    if "max_iter" in tsne_signature.parameters:
        tsne_kwargs["max_iter"] = int(config.reduction_tsne_n_iter)
    elif "n_iter" in tsne_signature.parameters:
        tsne_kwargs["n_iter"] = int(config.reduction_tsne_n_iter)
    else:
        raise RuntimeError("Unsupported sklearn.manifold.TSNE signature: missing max_iter/n_iter")
    tsne = TSNE(**tsne_kwargs)
    tsne_coords = tsne.fit_transform(distance_matrix)

    if umap_coords.shape != (len(all_rows), 2):
        raise RuntimeError(f"Unexpected UMAP output shape: {umap_coords.shape}")
    if tsne_coords.shape != (len(all_rows), 2):
        raise RuntimeError(f"Unexpected t-SNE output shape: {tsne_coords.shape}")

    points_path = base_dir / "combined_points_diversity.jsonl"
    distances_out_path = base_dir / "combined_edit_distance.npy"
    umap_out_path = base_dir / "combined_umap_diversity.npy"
    tsne_out_path = base_dir / "combined_tsne_diversity.npy"
    meta_path = base_dir / "reduction_meta_diversity.json"

    with points_path.open("w") as handle:
        for row, umap_point, tsne_point in zip(all_rows, umap_coords, tsne_coords):
            payload = {
                **row,
                "umap_x": float(umap_point[0]),
                "umap_y": float(umap_point[1]),
                "tsne_x": float(tsne_point[0]),
                "tsne_y": float(tsne_point[1]),
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    np.save(distances_out_path, distance_matrix.astype(np.float32, copy=False))
    np.save(umap_out_path, umap_coords.astype(np.float32, copy=False))
    np.save(tsne_out_path, tsne_coords.astype(np.float32, copy=False))
    meta_path.write_text(
        json.dumps(
            {
                "num_points": len(all_rows),
                "distance_metric": "1 - normalized_edit_similarity",
                "valid_only": True,
                "block_size": int(args.block_size),
                "tsne_perplexity": float(config.reduction_tsne_perplexity),
                "tsne_n_iter": int(config.reduction_tsne_n_iter),
                "umap_n_neighbors": int(config.reduction_umap_n_neighbors),
                "umap_min_dist": float(config.reduction_umap_min_dist),
                "seed": int(config.seed),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(points_path)


if __name__ == "__main__":
    main()
