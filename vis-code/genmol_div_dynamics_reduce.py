#!/usr/bin/env python3
"""Reduce GenMol de novo dynamics samples using training-time Tanimoto diversity geometry."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))
sys.path.append(str(REPO_ROOT / "vis-code"))

from genmol_div_dynamics_common import load_dynamics_config, output_dir_path
from rl_shared.hbd import _get_hbd_cpu_worker_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--block-size", type=int, default=128)
    return parser.parse_args()


def _load_model_records(records_path: Path) -> list[dict]:
    rows = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No rows found in {records_path}")
    return rows


def _load_valid_rows(config, base_dir: Path) -> list[dict]:
    all_rows: list[dict] = []
    for model in config.models:
        records_path = base_dir / "per_model" / model.model_id / "records.jsonl"
        if not records_path.is_file():
            raise FileNotFoundError(f"Missing records file: {records_path}")
        rows = _load_model_records(records_path)
        valid_rows = [row for row in rows if bool(row["is_valid"]) and row["smiles"] is not None]
        if not valid_rows:
            raise ValueError(f"No valid rows found for {model.model_id} in {records_path}")
        all_rows.extend(valid_rows)
    if not all_rows:
        raise ValueError("No valid rows found across all models")
    return all_rows


def _build_fingerprints(rows: list[dict]):
    from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = []
    for row in rows:
        smiles = str(row["smiles"]).strip()
        if not smiles:
            raise ValueError(f"Encountered empty SMILES in supposedly valid row: {row['model_id']}#{row['sample_index']}")
        mol = MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            raise ValueError(f"Failed to parse valid SMILES {smiles!r} for {row['model_id']}#{row['sample_index']}")
        fingerprints.append(generator.GetFingerprint(mol))
    return tuple(fingerprints)


def _compute_similarity_block(query_fingerprints, candidate_fingerprints):
    from rdkit import DataStructs

    block = np.empty((len(query_fingerprints), len(candidate_fingerprints)), dtype=np.float32)
    for row_idx, fingerprint in enumerate(query_fingerprints):
        block[row_idx, :] = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fingerprint, candidate_fingerprints),
            dtype=np.float32,
        )
    return block


def _compute_distance_matrix(fingerprints, *, block_size: int) -> np.ndarray:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    count = len(fingerprints)
    if count == 0:
        raise ValueError("fingerprints must be non-empty")
    worker_count = _get_hbd_cpu_worker_count()
    distance_matrix = np.empty((count, count), dtype=np.float32)
    block_ranges = [(start, min(start + block_size, count)) for start in range(0, count, block_size)]

    def _run_block(block_range):
        start, stop = block_range
        similarities = _compute_similarity_block(fingerprints[start:stop], fingerprints)
        if similarities.shape != (stop - start, count):
            raise RuntimeError(
                f"Unexpected similarity block shape for {block_range}: {similarities.shape} vs {(stop - start, count)}"
            )
        return start, stop, similarities

    if worker_count <= 1 or len(block_ranges) == 1:
        for block_range in block_ranges:
            start, stop, similarities = _run_block(block_range)
            distance_matrix[start:stop, :] = 1.0 - similarities
    else:
        with ThreadPoolExecutor(
            max_workers=min(worker_count, len(block_ranges)),
            thread_name_prefix="genmol_div_reduce",
        ) as executor:
            futures = [executor.submit(_run_block, block_range) for block_range in block_ranges]
            for future in futures:
                start, stop, similarities = future.result()
                distance_matrix[start:stop, :] = 1.0 - similarities

    np.fill_diagonal(distance_matrix, 0.0)
    max_asymmetry = float(np.max(np.abs(distance_matrix - distance_matrix.T)))
    if max_asymmetry > 1e-6:
        raise RuntimeError(f"Tanimoto distance matrix is not symmetric enough: max_asymmetry={max_asymmetry}")
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
    fingerprints = _build_fingerprints(all_rows)
    distance_matrix = _compute_distance_matrix(fingerprints, block_size=int(args.block_size))

    try:
        import umap
    except Exception as exc:
        raise ImportError("umap-learn is required for GenMol diversity dynamics UMAP reduction") from exc

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
    distances_out_path = base_dir / "combined_tanimoto_distance.npy"
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
                "distance_metric": "1 - Morgan-fingerprint Tanimoto similarity",
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
