#!/usr/bin/env python3
"""Generate progen2 dynamics samples, score them, and compute ESM-2 embeddings."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "src"))
sys.path.append(str(REPO_ROOT / "vis-code"))

from progen2.data.prompts import load_prompt_texts
from progen2.evaluation import classify_protein_sequence, nanmean
from progen2.modeling.wrapper import OfficialProGen2CausalLM
from progen2.rewards import CompositeProteinReward
from progen2.rl.policy import ProGen2Policy
from progen2.rewards.common import iter_chunks
from progen2_dynamics_common import DynamicsModelSpec, load_dynamics_config, model_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-id", required=True)
    return parser.parse_args()


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device(device_name)


def _cycle_prompt_batch(prompts: list[str], batch_size: int, start_index: int) -> list[str]:
    return [prompts[(start_index + offset) % len(prompts)] for offset in range(batch_size)]


def _default_reward_batch_size(config) -> int:
    return int(config.generation_prompt_batch_size * config.num_return_sequences)


def _collect_calibration_sequences(policy, prompts, config, seed: int) -> list[str]:
    collected: list[str] = []
    prompt_cursor = 0
    attempts = 0
    max_batches = max(
        8,
        math.ceil(config.reward_calibration_size / config.reward_calibration_prompt_batch_size) * 20,
    )
    while len(collected) < config.reward_calibration_size:
        if attempts >= max_batches:
            raise RuntimeError(
                "Failed to collect enough valid calibration sequences before hitting the maximum "
                f"attempt budget: collected={len(collected)} target={config.reward_calibration_size} "
                f"max_batches={max_batches}"
            )
        remaining = config.reward_calibration_size - len(collected)
        prompt_batch_size = min(config.reward_calibration_prompt_batch_size, remaining)
        prompt_batch = _cycle_prompt_batch(prompts, prompt_batch_size, prompt_cursor)
        prompt_cursor = (prompt_cursor + len(prompt_batch)) % len(prompts)
        rollout = policy.generate_rollouts(
            prompt_batch,
            num_return_sequences=1,
            max_new_tokens=config.max_new_tokens,
            top_p=config.top_p,
            temperature=config.temperature,
            seed=seed + attempts,
        )
        for sequence in rollout.protein_sequences:
            classification = classify_protein_sequence(sequence)
            if classification["is_valid"]:
                collected.append(classification["sequence"])
        attempts += 1
    return collected[: config.reward_calibration_size]


def _generate_rows(policy, prompts, config, seed: int) -> list[dict]:
    rows: list[dict] = []
    prompt_cursor = 0
    batch_index = 0
    while len(rows) < config.num_samples:
        prompt_batch = _cycle_prompt_batch(prompts, config.generation_prompt_batch_size, prompt_cursor)
        prompt_cursor = (prompt_cursor + len(prompt_batch)) % len(prompts)
        rollout = policy.generate_rollouts(
            prompt_batch,
            num_return_sequences=config.num_return_sequences,
            max_new_tokens=config.max_new_tokens,
            top_p=config.top_p,
            temperature=config.temperature,
            seed=seed + batch_index,
        )
        for prompt_text, decoded_text, raw_sequence in zip(
            rollout.prompt_texts,
            rollout.decoded_texts,
            rollout.protein_sequences,
        ):
            classification = classify_protein_sequence(raw_sequence)
            rows.append(
                {
                    "sample_index": len(rows),
                    "prompt_text": prompt_text,
                    "decoded_text": decoded_text,
                    "raw_sequence": raw_sequence,
                    "sequence": classification["sequence"],
                    "is_valid": classification["is_valid"],
                    "invalid_reason": classification["invalid_reason"],
                }
            )
            if len(rows) >= config.num_samples:
                break
        batch_index += 1
    if len(rows) != config.num_samples:
        raise RuntimeError(f"Generated {len(rows)} rows, expected {config.num_samples}")
    return rows


def _initialize_row_metrics(rows: list[dict]) -> None:
    for row in rows:
        row.update(
            {
                "naturalness_raw": None,
                "naturalness": 0.0,
                "foldability": 0.0,
                "stability_raw": None,
                "stability": 0.0,
                "solubility": 0.0,
                "liability_reward": 0.0,
                "developability": 0.0,
                "soft_reward": 0.0,
            }
        )


def _score_rows(rows: list[dict], reward_model: CompositeProteinReward) -> dict:
    _initialize_row_metrics(rows)
    valid_indices = [idx for idx, row in enumerate(rows) if row["is_valid"]]
    valid_sequences = [rows[idx]["sequence"] for idx in valid_indices]
    reward_metrics = {
        "reward_nat_mean": 0.0,
        "reward_fold_mean": 0.0,
        "reward_stab_mean": 0.0,
        "reward_dev_mean": 0.0,
        "reward_sol_mean": 0.0,
        "reward_liability_mean": 0.0,
        "reward_total_mean": 0.0,
    }
    if valid_sequences:
        details, reward_metrics = reward_model.score_details(valid_sequences)
        for detail_index, row_index in enumerate(valid_indices):
            rows[row_index]["naturalness_raw"] = float(details["naturalness_raw"][detail_index])
            rows[row_index]["naturalness"] = float(details["naturalness"][detail_index])
            rows[row_index]["foldability"] = float(details["foldability"][detail_index])
            rows[row_index]["stability_raw"] = float(details["stability_raw"][detail_index])
            rows[row_index]["stability"] = float(details["stability"][detail_index])
            rows[row_index]["solubility"] = float(details["solubility"][detail_index])
            rows[row_index]["liability_reward"] = float(details["liability_reward"][detail_index])
            rows[row_index]["developability"] = float(details["developability"][detail_index])
            rows[row_index]["soft_reward"] = float(details["total"][detail_index])

    valid_only_metrics = {f"{key}_valid_only": float(value) for key, value in reward_metrics.items()}
    return {
        "soft_reward_mean": float(sum(row["soft_reward"] for row in rows) / len(rows)),
        "valid_fraction": float(sum(1 for row in rows if row["is_valid"]) / len(rows)),
        "mean_valid_length": nanmean([len(row["sequence"]) for row in rows if row["sequence"]]),
        "naturalness_raw_mean_valid": nanmean([row["naturalness_raw"] for row in rows]),
        "stability_raw_mean_valid": nanmean([row["stability_raw"] for row in rows]),
        **valid_only_metrics,
    }


def _embedding_sequence_from_row(row: dict) -> str:
    candidate = row["sequence"] if row["sequence"] else str(row["raw_sequence"]).strip().upper()
    if not candidate:
        raise ValueError(f"sample_index={row['sample_index']} produced an empty sequence and cannot be embedded")
    return candidate


@torch.no_grad()
def _embed_rows_with_esm2(rows: list[dict], reward_model: CompositeProteinReward, batch_size: int) -> np.ndarray:
    scorer = reward_model.naturalness
    if scorer is None:
        raise RuntimeError("Naturalness scorer is required for ESM-2 embedding extraction")
    scorer._ensure_loaded()  # reuse the already provisioned ESM-2 asset in the reward stack
    if scorer.model is None or scorer.batch_converter is None:
        raise RuntimeError("Failed to initialize ESM-2 model for embeddings")
    sequences = [_embedding_sequence_from_row(row) for row in rows]
    model = scorer.model
    batch_converter = scorer.batch_converter
    layer = int(getattr(model, "num_layers", 33))
    all_embeddings: list[np.ndarray] = []
    for batch_start, batch_sequences in enumerate(iter_chunks(sequences, batch_size)):
        batch = [
            (str(batch_start * batch_size + row_idx), sequence)
            for row_idx, sequence in enumerate(batch_sequences)
        ]
        _, _, tokens = batch_converter(batch)
        tokens = tokens.to(scorer.device)
        outputs = model(tokens, repr_layers=[layer], return_contacts=False)
        representations = outputs["representations"][layer]
        for row_idx, sequence in enumerate(batch_sequences):
            residue_repr = representations[row_idx, 1:1 + len(sequence)]
            if residue_repr.numel() == 0:
                raise RuntimeError(
                    f"ESM-2 returned an empty representation for a non-empty sequence of length {len(sequence)}"
                )
            pooled = residue_repr.mean(dim=0).detach().cpu().numpy().astype(np.float32, copy=False)
            all_embeddings.append(pooled)
    embeddings = np.stack(all_embeddings, axis=0)
    if embeddings.shape[0] != len(rows):
        raise RuntimeError(f"Embedded {embeddings.shape[0]} rows, expected {len(rows)}")
    return embeddings


def _load_model_spec(config, model_id: str) -> DynamicsModelSpec:
    for model in config.models:
        if model.model_id == model_id:
            return model
    available = ", ".join(model.model_id for model in config.models)
    raise ValueError(f"Unknown model_id {model_id!r}; available: {available}")


def _build_reward_model(config, device: torch.device) -> CompositeProteinReward:
    return CompositeProteinReward(
        config.rewards,
        device=device,
        default_reward_batch_size=_default_reward_batch_size(config),
        reward_weights=None,
        always_compute_metrics=True,
    )


def main() -> None:
    args = parse_args()
    config = load_dynamics_config(args.config)
    model = _load_model_spec(config, args.model_id)
    device = resolve_device(config.device)
    prompts = load_prompt_texts(config.prompt_path)

    policy = ProGen2Policy(
        OfficialProGen2CausalLM(
            official_code_dir=config.official_code_dir,
            checkpoint_dir=model.checkpoint_dir,
            tokenizer_path=config.tokenizer_path,
            checkpoint_subdir=model.checkpoint_subdir,
            device=device,
            use_fp16=False,
            autocast_dtype=torch.bfloat16 if config.bf16 and device.type == "cuda" else None,
        ),
        trainable=False,
    )
    reward_model = _build_reward_model(config, device)
    calibration_sequences = _collect_calibration_sequences(
        policy,
        prompts,
        config,
        seed=config.seed + (10000 * model.slot_index),
    )
    calibration = reward_model.calibrate(calibration_sequences)
    rows = _generate_rows(
        policy,
        prompts,
        config,
        seed=config.seed + (1000 * model.slot_index),
    )
    metrics = _score_rows(rows, reward_model)
    embeddings = _embed_rows_with_esm2(rows, reward_model, batch_size=config.embedding_batch_size)

    output_dir = model_output_dir(config, model)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    embeddings_path = output_dir / "embeddings.npy"
    summary_path = output_dir / "summary.json"

    _ensure_parent_dir(records_path)
    with records_path.open("w") as handle:
        for row in rows:
            payload = {
                "model_id": model.model_id,
                "row_id": model.row_id,
                "row_title": model.row_title,
                "slot_index": model.slot_index,
                "slot_label": model.slot_label,
                "display_name": model.display_name,
                "checkpoint_dir": model.checkpoint_dir,
                "temperature": float(config.temperature),
                "top_p": float(config.top_p),
                "max_new_tokens": int(config.max_new_tokens),
                "embedding_model_name": config.embedding_model_name,
                **row,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    np.save(embeddings_path, embeddings)
    summary = {
        "model_id": model.model_id,
        "row_id": model.row_id,
        "row_title": model.row_title,
        "slot_index": model.slot_index,
        "slot_label": model.slot_label,
        "display_name": model.display_name,
        "checkpoint_dir": model.checkpoint_dir,
        "num_rows": len(rows),
        "embedding_dim": int(embeddings.shape[1]),
        "calibration": calibration,
        **metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(summary_path)


if __name__ == "__main__":
    main()
