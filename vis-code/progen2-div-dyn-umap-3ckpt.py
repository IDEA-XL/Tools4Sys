#!/usr/bin/env python3
"""Plot a 6-panel ProGen2 diversity UMAP figure using original, ckpt20, and ckpt100."""

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
    (0, 1, "Original \u2192 20 Steps"),
    (1, 5, "20 Steps \u2192 Final"),
)
SLOT_INDEX_TO_LABEL = {
    0: "Original",
    1: "20 Steps",
    5: "Final",
}
SLOT_INDEX_TO_COLOR = {
    0: "#7A9E3A",
    1: "#8E63B6",
    5: "#1F4E79",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--points-jsonl", default=None)
    parser.add_argument("--figure-output-dir", default=None)
    parser.add_argument("--output-name", default="progen2-div-dyn-umap-3ckpt.svg")
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
            "font.size": 15,
            "axes.labelsize": 18,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 1.3,
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


def _pair_rows(row_index: dict[tuple[str, int], list[dict]], row_id: str, left_slot: int, right_slot: int) -> list[dict]:
    left_rows = row_index[(row_id, left_slot)]
    right_rows = row_index[(row_id, right_slot)]
    if not left_rows or not right_rows:
        raise ValueError(f"Missing row/slot data for row_id={row_id}, pair=({left_slot}, {right_slot})")
    return left_rows + right_rows


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("#FBFBFA")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#111111")
        spine.set_linewidth(1.2)


def _plot_grid(rows: list[dict], output_path: Path) -> None:
    _configure_style()
    row_index = _build_index(rows)
    panel_rows: list[dict] = []
    for row_id, _ in ROW_ORDER:
        for left_slot, right_slot, _ in PAIR_COLUMNS:
            panel_rows.extend(_pair_rows(row_index, row_id, left_slot, right_slot))
    x_min, x_max, y_min, y_max = _finite_extent(panel_rows, "umap_x", "umap_y")

    fig, axes = plt.subplots(1, 6, figsize=(22.8, 4.8))
    fig.patch.set_facecolor("white")

    for group_idx, (row_id, _) in enumerate(ROW_ORDER):
        for pair_idx, (left_slot, right_slot, _) in enumerate(PAIR_COLUMNS):
            ax = axes[group_idx * 2 + pair_idx]
            _style_axis(ax)
            left_rows = row_index[(row_id, left_slot)]
            right_rows = row_index[(row_id, right_slot)]
            for slot_index, slot_rows in ((left_slot, left_rows), (right_slot, right_rows)):
                color = SLOT_INDEX_TO_COLOR[slot_index]
                ax.scatter(
                    [float(row["umap_x"]) for row in slot_rows],
                    [float(row["umap_y"]) for row in slot_rows],
                    s=22,
                    c=color,
                    marker="o",
                    alpha=0.76,
                    linewidths=0.0,
                    rasterized=True,
                )
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

    color_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SLOT_INDEX_TO_COLOR[slot_index], markersize=9, label=SLOT_INDEX_TO_LABEL[slot_index])
        for slot_index in (0, 1, 5)
    ]
    fig.legend(
        handles=color_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, -0.02),
        ncol=3,
        frameon=False,
        columnspacing=1.6,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.018, right=0.995, top=0.77, bottom=0.14, wspace=0.06)

    for group_idx, (_, row_title) in enumerate(ROW_ORDER):
        left_ax = axes[group_idx * 2]
        right_ax = axes[group_idx * 2 + 1]
        left_box = left_ax.get_position()
        right_box = right_ax.get_position()
        x_center = 0.5 * (left_box.x0 + right_box.x1)
        fig.text(
            x_center,
            left_box.y1 + 0.080,
            row_title,
            ha="center",
            va="bottom",
            fontsize=17,
            fontweight="semibold",
            color="#222222",
        )

    for axis_idx, ax in enumerate(axes):
        _, _, pair_label = PAIR_COLUMNS[axis_idx % 2]
        box = ax.get_position()
        fig.text(
            0.5 * (box.x0 + box.x1),
            box.y1 + 0.026,
            pair_label,
            ha="center",
            va="bottom",
            fontsize=12,
            color="#333333",
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
    output_path = figure_dir / str(args.output_name)
    _plot_grid(rows, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
