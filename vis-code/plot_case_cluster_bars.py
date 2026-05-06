#!/usr/bin/env python3
"""Plot clustered high-utility ProGen2 case-study bars."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main_pareto_common import COLOR_GRPO, COLOR_MEMORY, COLOR_ORIGINAL, COLOR_SGRPO


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "figs" / "case-gallery" / "progen2_case_manifest_cdhit06_softgt0.8.json"
OUTPUT_PATH = REPO_ROOT / "figs" / "case.svg"

MODEL_ORDER = (
    "Memory-Assisted GRPO",
    "GRPO",
    "Original",
    "SGRPO",
)

MODEL_COLORS = {
    "Original": COLOR_ORIGINAL,
    "GRPO": COLOR_GRPO,
    "SGRPO": COLOR_SGRPO,
    "Memory-Assisted GRPO": COLOR_MEMORY,
}

BAR_WIDTHS = {
    "Memory-Assisted GRPO": 0.82,
    "GRPO": 0.58,
    "Original": 0.22,
    "SGRPO": 0.06,
}

WIDTH_RATIOS = {
    "Memory-Assisted GRPO": 1.4,
    "GRPO": 1.7,
    "Original": 4.5,
    "SGRPO": 10.5,
}


def load_manifest(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def choose_ticks(cluster_ids: list[int], max_ticks: int) -> list[int]:
    if not cluster_ids:
        raise ValueError("Expected at least one cluster id")
    if len(cluster_ids) <= max_ticks:
        return cluster_ids
    positions = []
    last_index = len(cluster_ids) - 1
    for tick_index in range(max_ticks):
        if tick_index == max_ticks - 1:
            idx = last_index
        else:
            idx = round((last_index * tick_index) / float(max_ticks - 1))
        positions.append(cluster_ids[idx])
    deduped = []
    seen = set()
    for value in positions:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    if deduped[-1] != cluster_ids[-1]:
        deduped[-1] = cluster_ids[-1]
    return deduped


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def main() -> None:
    apply_publication_style()
    manifest = load_manifest(MANIFEST_PATH)
    counts_by_model = manifest["per_model_cluster_counts"]

    width_ratios = [WIDTH_RATIOS[model] for model in MODEL_ORDER]
    fig, axes = plt.subplots(
        1,
        len(MODEL_ORDER),
        figsize=(20, 4.6),
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.28},
        sharey=True,
    )

    max_count = max(
        item["count"]
        for model in MODEL_ORDER
        for item in counts_by_model[model]
    )

    for axis, model in zip(axes, MODEL_ORDER):
        series = counts_by_model[model]
        cluster_ids = [int(item["cluster_id"]) for item in series]
        counts = [int(item["count"]) for item in series]

        axis.bar(
            cluster_ids,
            counts,
            width=BAR_WIDTHS[model],
            color=MODEL_COLORS[model],
            edgecolor=MODEL_COLORS[model],
            linewidth=0.0,
        )

        axis.set_ylim(0, max_count + 2)
        left = cluster_ids[0] - max(0.7, BAR_WIDTHS[model] * 1.15)
        right = cluster_ids[-1] + max(0.7, BAR_WIDTHS[model] * 1.15)
        axis.set_xlim(left, right)
        axis.set_xticks(choose_ticks(cluster_ids, max_ticks=7))
        axis.set_xlabel("")
        axis.tick_params(axis="x", length=4, width=0.8)
        axis.tick_params(axis="y", length=4, width=0.8)
        axis.text(
            0.03,
            0.95,
            model,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=18,
        )

    axes[0].set_ylabel("Protein Count")
    fig.supxlabel("Cluster ID", y=0.03, fontsize=20)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, format="svg")
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
