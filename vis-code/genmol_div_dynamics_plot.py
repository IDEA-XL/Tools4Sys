#!/usr/bin/env python3
"""Plot 3x10 GenMol de novo diversity-geometry dynamics scatter grids for UMAP and t-SNE."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "vis-code"))

from genmol_div_dynamics_common import load_dynamics_config, output_dir_path


SLOT_COLORS = (
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#B07AA1",
    "#76B7B2",
    "#EDC948",
    "#9C755F",
    "#FF9DA7",
    "#BAB0AC",
    "#2F4B7C",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--points-jsonl", default=None)
    parser.add_argument("--figure-output-dir", default=None)
    parser.add_argument("--umap-output-name", default="genmol-div-dyn-umap.pdf")
    parser.add_argument("--tsne-output-name", default="genmol-div-dyn-tsne.pdf")
    return parser.parse_args()


def _load_points(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.1,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def _build_row_order(config) -> tuple[tuple[str, str], ...]:
    row_pairs = []
    seen = set()
    for model in config.models:
        key = (model.row_id, model.row_title)
        if key in seen:
            continue
        seen.add(key)
        row_pairs.append(key)
    return tuple(row_pairs)


def _build_slot_labels(config) -> tuple[str, ...]:
    labels_by_index = {}
    for model in config.models:
        slot_index = int(model.slot_index)
        slot_label = str(model.slot_label)
        existing = labels_by_index.get(slot_index)
        if existing is not None and existing != slot_label:
            raise ValueError(f"Conflicting slot labels for slot_index={slot_index}: {existing!r} vs {slot_label!r}")
        labels_by_index[slot_index] = slot_label
    ordered_indices = sorted(labels_by_index)
    expected = list(range(len(ordered_indices)))
    if ordered_indices != expected:
        raise ValueError(f"slot indices must be contiguous starting at 0, got {ordered_indices}")
    return tuple(labels_by_index[idx] for idx in ordered_indices)


def _build_index(rows: list[dict]) -> dict[tuple[str, int], list[dict]]:
    index: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        index[(str(row["row_id"]), int(row["slot_index"]))].append(row)
    return index


def _finite_extent(rows: list[dict], x_key: str, y_key: str) -> tuple[float, float, float, float]:
    xs = [float(row[x_key]) for row in rows]
    ys = [float(row[y_key]) for row in rows]
    if not all(math.isfinite(value) for value in xs + ys):
        raise ValueError(f"Non-finite coordinates found for {x_key}/{y_key}")
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = max((x_max - x_min) * 0.06, 1e-3)
    y_pad = max((y_max - y_min) * 0.06, 1e-3)
    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


def _plot_grid(rows: list[dict], config, x_key: str, y_key: str, output_path: Path) -> None:
    _configure_style()
    row_order = _build_row_order(config)
    slot_labels = _build_slot_labels(config)
    if len(slot_labels) > len(SLOT_COLORS):
        raise ValueError(f"Need {len(slot_labels)} slot colors, only have {len(SLOT_COLORS)}")
    pair_columns = tuple((idx, idx + 1) for idx in range(len(slot_labels) - 1))
    row_index = _build_index(rows)
    x_min, x_max, y_min, y_max = _finite_extent(rows, x_key, y_key)

    fig, axes = plt.subplots(len(row_order), len(pair_columns), figsize=(36.0, 10.8))
    fig.patch.set_facecolor("white")

    for row_idx, (row_id, row_title) in enumerate(row_order):
        for col_idx, (left_slot, right_slot) in enumerate(pair_columns):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("#FBFBFA")
            left_rows = row_index[(row_id, left_slot)]
            right_rows = row_index[(row_id, right_slot)]
            if not left_rows or not right_rows:
                raise ValueError(f"Missing row/slot data for row_id={row_id}, pair=({left_slot}, {right_slot})")
            for slot_index, slot_rows in ((left_slot, left_rows), (right_slot, right_rows)):
                color = SLOT_COLORS[slot_index]
                threshold = float(config.soft_reward_marker_threshold)
                circle_rows = [row for row in slot_rows if float(row["soft_reward"]) <= threshold]
                triangle_rows = [row for row in slot_rows if float(row["soft_reward"]) > threshold]
                if circle_rows:
                    ax.scatter(
                        [float(row[x_key]) for row in circle_rows],
                        [float(row[y_key]) for row in circle_rows],
                        s=16,
                        c=color,
                        marker="o",
                        alpha=0.72,
                        linewidths=0.0,
                        rasterized=True,
                    )
                if triangle_rows:
                    ax.scatter(
                        [float(row[x_key]) for row in triangle_rows],
                        [float(row[y_key]) for row in triangle_rows],
                        s=22,
                        c=color,
                        marker="^",
                        alpha=0.82,
                        linewidths=0.0,
                        rasterized=True,
                    )
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.spines["left"].set_color("#CCCCCC")
            ax.spines["bottom"].set_color("#CCCCCC")

    color_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markersize=8.5, label=label)
        for color, label in zip(SLOT_COLORS[: len(slot_labels)], slot_labels)
    ]
    shape_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#666666",
            markersize=8,
            label=f"Soft reward ≤ {config.soft_reward_marker_threshold}",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="#666666",
            markersize=9,
            label=f"Soft reward > {config.soft_reward_marker_threshold}",
        ),
    ]
    fig.legend(
        handles=color_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 1.015),
        ncol=6,
        frameon=False,
        columnspacing=1.1,
        handletextpad=0.45,
    )
    fig.legend(
        handles=shape_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, -0.01),
        ncol=2,
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.6,
    )
    fig.subplots_adjust(left=0.03, right=0.997, top=0.88, bottom=0.08, wspace=0.06, hspace=0.18)

    for row_idx, (_, row_title) in enumerate(row_order):
        first_ax = axes[row_idx, 0]
        bbox = first_ax.get_position()
        fig.text(
            bbox.x0,
            bbox.y1 + 0.028,
            row_title,
            ha="left",
            va="bottom",
            fontsize=18,
            fontweight="semibold",
            color="#222222",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_dynamics_config(args.config)
    data_dir = output_dir_path(config)
    figure_dir = Path(args.figure_output_dir) if args.figure_output_dir else Path(config.figure_output_dir)
    points_path = Path(args.points_jsonl) if args.points_jsonl else data_dir / "combined_points_diversity.jsonl"
    rows = _load_points(points_path)
    umap_output_path = figure_dir / str(args.umap_output_name)
    tsne_output_path = figure_dir / str(args.tsne_output_name)
    _plot_grid(rows, config, "umap_x", "umap_y", umap_output_path)
    _plot_grid(rows, config, "tsne_x", "tsne_y", tsne_output_path)
    print(umap_output_path)
    print(tsne_output_path)


if __name__ == "__main__":
    main()
