#!/usr/bin/env python3
"""Combine progen2 dynamics embeddings and run UMAP / t-SNE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "vis-code"))

from progen2_dynamics_common import load_dynamics_config, model_output_dir, output_dir_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _load_model_records(records_path: Path) -> list[dict]:
    rows = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No rows found in {records_path}")
    return rows


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def main() -> None:
    args = parse_args()
    config = load_dynamics_config(args.config)
    base_dir = output_dir_path(config)
    base_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    all_embeddings: list[np.ndarray] = []
    for model in config.models:
        model_dir = model_output_dir(config, model)
        records_path = model_dir / "records.jsonl"
        embeddings_path = model_dir / "embeddings.npy"
        if not records_path.is_file():
            raise FileNotFoundError(f"Missing records file: {records_path}")
        if not embeddings_path.is_file():
            raise FileNotFoundError(f"Missing embeddings file: {embeddings_path}")
        rows = _load_model_records(records_path)
        embeddings = np.load(embeddings_path)
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings in {embeddings_path}, got shape {embeddings.shape}")
        if embeddings.shape[0] != len(rows):
            raise ValueError(
                f"Embedding row mismatch for {model.model_id}: {embeddings.shape[0]} vs {len(rows)}"
            )
        all_rows.extend(rows)
        all_embeddings.append(embeddings.astype(np.float32, copy=False))

    embedding_matrix = np.concatenate(all_embeddings, axis=0)
    normalized_embeddings = _l2_normalize(embedding_matrix)

    pca_dim = min(
        int(config.reduction_pca_components),
        int(normalized_embeddings.shape[0] - 1),
        int(normalized_embeddings.shape[1]),
    )
    if pca_dim < 2:
        raise ValueError(f"PCA dimension collapsed below 2: {pca_dim}")
    pca = PCA(n_components=pca_dim, random_state=int(config.seed))
    pca_embeddings = pca.fit_transform(normalized_embeddings)

    try:
        import umap
    except Exception as exc:
        raise ImportError("umap-learn is required for progen2 dynamics UMAP reduction") from exc

    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=int(config.reduction_umap_n_neighbors),
        min_dist=float(config.reduction_umap_min_dist),
        metric="euclidean",
        random_state=int(config.seed),
        low_memory=True,
    )
    umap_coords = umap_model.fit_transform(pca_embeddings)

    tsne = TSNE(
        n_components=2,
        perplexity=float(config.reduction_tsne_perplexity),
        init="pca",
        learning_rate="auto",
        n_iter=int(config.reduction_tsne_n_iter),
        metric="euclidean",
        random_state=int(config.seed),
        verbose=1,
    )
    tsne_coords = tsne.fit_transform(pca_embeddings)

    if umap_coords.shape != (len(all_rows), 2):
        raise RuntimeError(f"Unexpected UMAP output shape: {umap_coords.shape}")
    if tsne_coords.shape != (len(all_rows), 2):
        raise RuntimeError(f"Unexpected t-SNE output shape: {tsne_coords.shape}")

    points_path = base_dir / "combined_points.jsonl"
    embeddings_out_path = base_dir / "combined_embeddings.npy"
    pca_out_path = base_dir / "combined_pca_embeddings.npy"
    umap_out_path = base_dir / "combined_umap.npy"
    tsne_out_path = base_dir / "combined_tsne.npy"
    meta_path = base_dir / "reduction_meta.json"

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

    np.save(embeddings_out_path, normalized_embeddings.astype(np.float32, copy=False))
    np.save(pca_out_path, pca_embeddings.astype(np.float32, copy=False))
    np.save(umap_out_path, umap_coords.astype(np.float32, copy=False))
    np.save(tsne_out_path, tsne_coords.astype(np.float32, copy=False))
    meta_path.write_text(
        json.dumps(
            {
                "num_points": len(all_rows),
                "embedding_dim": int(normalized_embeddings.shape[1]),
                "pca_dim": int(pca_dim),
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
