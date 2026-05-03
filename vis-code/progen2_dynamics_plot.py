#!/usr/bin/env python3
"""Plot 3x5 progen2 dynamics scatter grids for UMAP and t-SNE."""

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

from progen2_dynamics_common import load_dynamics_config, output_dir_path


ROW_ORDER = (
    ("grpo", "GRPO"),
    ("grpo_hbd", "Memory-Assisted GRPO"),
    ("sgrpo_gw08_rewardsum_loo", "SGRPO"),
)
PAIR_COLUMNS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
)
SLOT_LABELS = ("Original", "Ckpt 20", "Ckpt 40", "Ckpt 60", "Ckpt 80", "Ckpt 100")
SLOT_COLORS = ("#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1", "#76B7B2")
SOFT_REWARD_THRESHOLD = 0.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--points-jsonl", default=None)
    parser.add_argument("--figure-output-dir", default=None)
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


def _build_index(rows: list[dict]) -> dict[tuple[str, int], list[dict]]:
    index: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        row_id = str(row["row_id"])
        slot_index = int(row["slot_index"])
        index[(row_id, slot_index)].append(row)
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


def _plot_grid(rows: list[dict], x_key: str, y_key: str, output_path: Path) -> None:
    _configure_style()
    row_index = _build_index(rows)
    x_min, x_max, y_min, y_max = _finite_extent(rows, x_key, y_key)

    fig, axes = plt.subplots(3, 5, figsize=(18.8, 10.6))
    fig.patch.set_facecolor("white")

    for row_idx, (row_id, row_title) in enumerate(ROW_ORDER):
        for col_idx, (left_slot, right_slot) in enumerate(PAIR_COLUMNS):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("#FBFBFA")
            left_rows = row_index[(row_id, left_slot)]
            right_rows = row_index[(row_id, right_slot)]
            if not left_rows or not right_rows:
                raise ValueError(f"Missing row/slot data for row_id={row_id}, pair=({left_slot}, {right_slot})")
            pair_rows = left_rows + right_rows
            for slot_index, slot_rows in ((left_slot, left_rows), (right_slot, right_rows)):
                color = SLOT_COLORS[slot_index]
                circle_rows = [row for row in slot_rows if float(row["soft_reward"]) <= SOFT_REWARD_THRESHOLD]
                triangle_rows = [row for row in slot_rows if float(row["soft_reward"]) > SOFT_REWARD_THRESHOLD]
                if circle_rows:
                    ax.scatter(
                        [float(row[x_key]) for row in circle_rows],
                        [float(row[y_key]) for row in circle_rows],
                        s=18,
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
                        s=26,
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
            if row_idx == 0:
                left_label = SLOT_LABELS[left_slot].replace("Ckpt ", "")
                right_label = SLOT_LABELS[right_slot].replace("Ckpt ", "")
                ax.text(
                    0.02,
                    1.04,
                    f"{left_label} vs {right_label}",
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=14,
                    color="#444444",
                )
            if col_idx == 0:
                ax.text(
                    0.00,
                    1.14,
                    row_title,
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=19,
                    fontweight="semibold",
                    color="#222222",
                )

    color_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markersize=9, label=label)
        for color, label in zip(SLOT_COLORS, SLOT_LABELS)
    ]
    shape_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666", markersize=8, label="Soft reward ≤ 0.6"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#666666", markersize=9, label="Soft reward > 0.6"),
    ]
    fig.legend(
        handles=color_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 1.02),
        ncol=6,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
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
    fig.subplots_adjust(left=0.035, right=0.995, top=0.90, bottom=0.08, wspace=0.06, hspace=0.20)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_dynamics_config(args.config)
    data_dir = output_dir_path(config)
    figure_dir = Path(args.figure_output_dir) if args.figure_output_dir else Path(config.figure_output_dir)
    points_path = Path(args.points_jsonl) if args.points_jsonl else data_dir / "combined_points.jsonl"
    rows = _load_points(points_path)
    _plot_grid(rows, "umap_x", "umap_y", figure_dir / "dyn-umap.pdf")
    _plot_grid(rows, "tsne_x", "tsne_y", figure_dir / "dyn-tsne.pdf")
    print(figure_dir / "dyn-umap.pdf")
    print(figure_dir / "dyn-tsne.pdf")


if __name__ == "__main__":
    main()
