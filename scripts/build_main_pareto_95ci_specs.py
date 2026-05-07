#!/usr/bin/env python3
"""Generate remote evaluation specs for the 4 additional main-Pareto CI runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_REMOTE_ROOT = Path("/public/home/xinwuye/ai4s-tool-joint-train/genmol")
RUN_OUTPUT_ROOT = Path("/public/home/xinwuye/ai4s-tool-joint-train/runs/main_pareto_95ci")
SPEC_OUTPUT_ROOT = RUN_OUTPUT_ROOT / "specs"
SEEDS = (43, 44, 45, 46)
PAIRED_SWEEP = (
    {"randomness": 0.1, "generation_temperature": 0.5},
    {"randomness": 0.3, "generation_temperature": 0.8},
    {"randomness": 0.5, "generation_temperature": 1.1},
    {"randomness": 0.7, "generation_temperature": 1.4},
    {"randomness": 0.9, "generation_temperature": 1.7},
    {"randomness": 1.0, "generation_temperature": 2.0},
)
PROGEN2_TEMPERATURES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)


DE_NOVO_EXPERIMENTS = (
    {
        "name": "original_genmol_v2",
        "display_name": "Original GenMol v2",
        "checkpoint_path": "/public/home/xinwuye/ai4s-tool-joint-train/genmol/checkpoints/genmol_v2_v1.0/model_v2.ckpt",
    },
    {
        "name": "genmol_denovo_grpo_2000",
        "display_name": "GenMol De Novo GRPO 2000",
        "checkpoint_path": "/public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_denovo/cpgrpo_denovo_ng512_bs1024_lr5e-5_beta5e-3_ni1_ms2000_20260422_161812/checkpoint-002000/model.ckpt",
    },
    {
        "name": "genmol_denovo_sgrpo_rewardsum_loo_2000",
        "display_name": "GenMol De Novo SGRPO RewardSum LOO 2000",
        "checkpoint_path": "/public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_denovo/cpgrpo_denovo_sgrpo_ng64_sg8_bs1024_lr5e-5_beta5e-3_gw09_rewardsum_loo_ms2000_20260426_115639/checkpoint-002000/model.ckpt",
    },
    {
        "name": "genmol_denovo_grpo_hbd_2000",
        "display_name": "GenMol De Novo GRPO HBD 2000",
        "checkpoint_path": "/public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_denovo/cpgrpo_denovo_ng512_bs1024_lr5e-5_beta5e-3_ni1_ms2000_hbd_st09_sc04_20260503_141949/checkpoint-002000/model.ckpt",
    },
)

MMGENMOL_TASK_SPECS = (
    (
        "original_5500",
        "/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_supervised_8gpu/20260416_151741/checkpoints/5500.ckpt",
    ),
    (
        "grpo_unidock_1000",
        "/public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_denovo_pocket_prefix/cpgrpo_denovo_pocket_prefix_ng192_bs384_lr5e-5_beta5e-3_ni1_q03_sa02_unidock05_20260430_192150/checkpoint-001000/model.ckpt",
    ),
    (
        "sgrpo_unidock_rewardsum_loo_1000",
        "/public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_denovo_pocket_prefix/cpgrpo_denovo_pocket_prefix_sgrpo_ng24_sg8_bs384_lr5e-5_beta5e-3_gw09_q03_sa02_unidock05_rewardsum_loo_20260501_160306/checkpoint-001000/model.ckpt",
    ),
)

PROGEN2_EXPERIMENTS = (
    {
        "name": "original",
        "display_name": "Original",
        "checkpoint_dir": "/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_official/checkpoints/progen2-small",
        "naturalness": 0.25,
        "foldability": 0.30,
        "stability": 0.20,
        "developability": 0.25,
    },
    {
        "name": "grpo_step100",
        "display_name": "GRPO 100",
        "checkpoint_dir": "/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_sgrpo/progen2_grpo_ng96_bs2_len256_rbs16_slurm52245/checkpoint-000100",
        "naturalness": 0.25,
        "foldability": 0.30,
        "stability": 0.20,
        "developability": 0.25,
    },
    {
        "name": "sgrpo_gw08_rewardsum_loo_step100",
        "display_name": "SGRPO gw0.8 RewardSum LOO 100",
        "checkpoint_dir": "/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_sgrpo/progen2_sgrpo_ng12_sg8_bs2_len256_rbs16_gw08_rewardsum_loo_slurm53602/checkpoint-000100",
        "naturalness": 0.25,
        "foldability": 0.30,
        "stability": 0.20,
        "developability": 0.25,
    },
    {
        "name": "grpo_hbd_step100",
        "display_name": "GRPO HBD 100",
        "checkpoint_dir": "/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_sgrpo/progen2_grpo_ng96_bs2_len256_rbs16_hbd_slurm55873/checkpoint-000100",
        "naturalness": 0.25,
        "foldability": 0.30,
        "stability": 0.20,
        "developability": 0.25,
    },
)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _dump_yaml_like_mapping(mapping: dict, indent: int = 0) -> list[str]:
    lines: list[str] = []
    prefix = " " * indent
    for key, value in mapping.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.extend(_dump_yaml_like_mapping(value, indent=indent + 2))
        elif isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            for item in value:
                if isinstance(item, dict):
                    lines.append(f"{prefix}  -")
                    for child_key, child_value in item.items():
                        lines.extend(_dump_yaml_like_mapping({child_key: child_value}, indent=indent + 4))
                else:
                    lines.append(f"{prefix}  - {json.dumps(item)}")
        elif value is None:
            lines.append(f"{prefix}{key}: null")
        elif isinstance(value, bool):
            lines.append(f"{prefix}{key}: {'true' if value else 'false'}")
        elif isinstance(value, (int, float)):
            lines.append(f"{prefix}{key}: {value}")
        else:
            lines.append(f"{prefix}{key}: {value}")
    return lines


def _build_denovo_config(seed: int) -> str:
    run_root = RUN_OUTPUT_ROOT / "denovo" / f"seed{seed}"
    payload = {
        "output_markdown_path": str(run_root / "aggregate" / "denovo_paired.md"),
        "output_json_path": str(run_root / "aggregate" / "denovo_paired.json"),
        "output_qed_diversity_plot_path": str(run_root / "aggregate" / "qed_vs_diversity.png"),
        "output_sa_score_diversity_plot_path": str(run_root / "aggregate" / "sa_score_vs_diversity.png"),
        "output_soft_reward_diversity_plot_path": str(run_root / "aggregate" / "soft_reward_vs_diversity.png"),
        "output_rows_path": str(run_root / "aggregate" / "denovo_paired.rows.jsonl"),
        "seed": seed,
        "bf16": True,
        "device": "cuda",
        "num_samples": 1000,
        "generation_batch_size": 2048,
        "randomness_temperature_pairs": list(PAIRED_SWEEP),
        "min_add_len": 60,
        "max_completion_length": None,
        "experiments": list(DE_NOVO_EXPERIMENTS),
    }
    return "\n".join(_dump_yaml_like_mapping(payload)) + "\n"


def _build_mmgenmol_tasks(seed: int) -> str:
    lines = [
        "\t".join(
            [
                "task_id",
                "model_name",
                "sweep_type",
                "sweep_value",
                "randomness",
                "temperature",
                "checkpoint_path",
                "output_path",
            ]
        )
    ]
    task_id = 0
    for model_name, checkpoint_path in MMGENMOL_TASK_SPECS:
        for sweep_index, pair in enumerate(PAIRED_SWEEP, start=1):
            output_path = (
                RUN_OUTPUT_ROOT
                / "mmgenmol"
                / f"seed{seed}"
                / "generation"
                / model_name
                / f"paired_{sweep_index}"
                / "generated.rows.jsonl"
            )
            lines.append(
                "\t".join(
                    [
                        str(task_id),
                        model_name,
                        "paired",
                        str(sweep_index),
                        f"{pair['randomness']:.1f}",
                        f"{pair['generation_temperature']:.1f}",
                        checkpoint_path,
                        str(output_path),
                    ]
                )
            )
            task_id += 1
    return "\n".join(lines) + "\n"


def _build_progen2_config(seed: int) -> str:
    run_root = RUN_OUTPUT_ROOT / "progen2" / f"seed{seed}"
    payload = {
        "tasks_path": str(run_root / "specs" / "progen2_temperature_tasks.tsv"),
        "generation_output_root": str(run_root / "generation"),
        "foldability_output_root": str(run_root / "foldability"),
        "developability_output_root": str(run_root / "developability"),
        "diversity_output_root": str(run_root / "diversity"),
        "packed_naturalness_scores_path": str(run_root / "naturalness" / "naturalness.rows.jsonl"),
        "packed_stability_scores_path": str(run_root / "stability" / "stability.rows.jsonl"),
        "output_markdown_path": str(run_root / "aggregate" / "progen2_temperature_sweep.md"),
        "output_json_path": str(run_root / "aggregate" / "progen2_temperature_sweep.json"),
        "output_rows_path": str(run_root / "aggregate" / "progen2_temperature_sweep.rows.jsonl"),
        "output_naturalness_diversity_plot_path": str(run_root / "aggregate" / "naturalness_vs_diversity.png"),
        "output_foldability_diversity_plot_path": str(run_root / "aggregate" / "foldability_vs_diversity.png"),
        "output_stability_diversity_plot_path": str(run_root / "aggregate" / "stability_vs_diversity.png"),
        "output_developability_diversity_plot_path": str(run_root / "aggregate" / "developability_vs_diversity.png"),
        "output_soft_reward_diversity_plot_path": str(run_root / "aggregate" / "soft_reward_vs_diversity.png"),
        "official_code_dir": "/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_official",
        "tokenizer_path": "/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_official/tokenizer.json",
        "prompt_path": "/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_official/prompts_unconditional.txt",
        "seed": seed,
        "bf16": False,
        "device": "cuda",
        "num_samples": 512,
        "generation_prompt_batch_size": 1,
        "num_return_sequences": 512,
        "max_new_tokens": 256,
        "top_p": 0.95,
        "temperature_values": list(PROGEN2_TEMPERATURES),
        "calibration_temperature": 0.8,
        "reward_calibration_size": 256,
        "reward_calibration_prompt_batch_size": 128,
        "rewards": {
            "naturalness": {
                "model_name": "esm2_t33_650M_UR50D",
                "device": "cuda",
                "batch_size": 4096,
            },
            "foldability": {
                "device": "cuda",
                "batch_size": 64,
            },
            "stability": {
                "model_name_or_path": "/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/temberture_official",
                "base_model_name_or_path": "/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/prot_bert_bfd",
                "device": "cuda",
                "batch_size": 8192,
            },
            "developability": {
                "model_name_or_path": "/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/proteinsol_official",
                "device": "cpu",
                "batch_size": 24,
                "num_workers": 64,
            },
        },
        "experiments": list(PROGEN2_EXPERIMENTS),
    }
    return "\n".join(_dump_yaml_like_mapping(payload)) + "\n"


def _build_progen2_tasks(seed: int) -> str:
    fieldnames = [
        "task_id",
        "experiment",
        "display_name",
        "checkpoint_dir",
        "checkpoint_subdir",
        "naturalness_weight",
        "foldability_weight",
        "stability_weight",
        "developability_weight",
        "temperature",
        "generation_rows_path",
        "foldability_scores_path",
        "developability_scores_path",
        "diversity_scores_path",
    ]
    lines = ["\t".join(fieldnames)]
    run_root = RUN_OUTPUT_ROOT / "progen2" / f"seed{seed}"
    task_id = 0
    for experiment in PROGEN2_EXPERIMENTS:
        for temperature in PROGEN2_TEMPERATURES:
            leaf = f"temperature_{temperature:.1f}"
            lines.append(
                "\t".join(
                    [
                        str(task_id),
                        experiment["name"],
                        experiment["display_name"],
                        experiment["checkpoint_dir"],
                        "",
                        f"{experiment['naturalness']:.2f}",
                        f"{experiment['foldability']:.2f}",
                        f"{experiment['stability']:.2f}",
                        f"{experiment['developability']:.2f}",
                        f"{temperature:.1f}",
                        str(run_root / "generation" / experiment["name"] / leaf / "generated.rows.jsonl"),
                        str(run_root / "foldability" / experiment["name"] / leaf / "foldability.rows.jsonl"),
                        str(run_root / "developability" / experiment["name"] / leaf / "developability.rows.jsonl"),
                        str(run_root / "diversity" / experiment["name"] / leaf / "diversity.json"),
                    ]
                )
            )
            task_id += 1
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-root", type=Path, default=SPEC_OUTPUT_ROOT)
    args = parser.parse_args()

    spec_root: Path = args.spec_root
    manifest = {"repo_root": str(REPO_REMOTE_ROOT), "run_output_root": str(RUN_OUTPUT_ROOT), "seeds": list(SEEDS)}

    for seed in SEEDS:
        _write_text(spec_root / "denovo" / f"seed{seed}.yaml", _build_denovo_config(seed))
        _write_text(spec_root / "mmgenmol" / f"seed{seed}.tsv", _build_mmgenmol_tasks(seed))
        _write_text(spec_root / "progen2" / f"seed{seed}.yaml", _build_progen2_config(seed))
        _write_text(spec_root / "progen2" / f"seed{seed}_tasks.tsv", _build_progen2_tasks(seed))

    _write_text(spec_root / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(spec_root)


if __name__ == "__main__":
    main()
