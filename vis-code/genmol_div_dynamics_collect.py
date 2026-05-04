#!/usr/bin/env python3
"""Generate GenMol de novo dynamics samples and score them with the training-time soft reward."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "src"))
sys.path.append(str(REPO_ROOT / "vis-code"))

from genmol.rl.policy import GenMolCpGRPOPolicy
from genmol.rl.reward import MolecularReward
from genmol.rl.specs import sample_group_specs
from genmol_div_dynamics_common import load_dynamics_config, model_output_dir


def _log(message: str) -> None:
    print(f"[genmol-div-dynamics] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--model-index", type=int, default=None)
    return parser.parse_args()


def _resolve_model(config, *, model_id: str | None, model_index: int | None):
    if (model_id is None) == (model_index is None):
        raise ValueError("Exactly one of --model-id or --model-index must be provided")
    if model_id is not None:
        for model in config.models:
            if model.model_id == model_id:
                return model
        raise ValueError(f"Unknown model_id {model_id!r}")
    if not 0 <= int(model_index) < len(config.models):
        raise ValueError(f"model_index out of range: {model_index} vs {len(config.models)} models")
    return config.models[int(model_index)]


def _build_rows(config, model) -> list[dict]:
    _log(f"loading policy from {model.checkpoint_path}")
    policy = GenMolCpGRPOPolicy(
        checkpoint_path=model.checkpoint_path,
        device=config.device,
        bf16=config.bf16,
        trainable=False,
    )
    reward_model = MolecularReward(
        reward_weights=config.reward_weights,
        always_compute_metrics=True,
    )
    try:
        _log("sampling rollout specs")
        specs = sample_group_specs(
            num_groups=config.num_samples,
            generation_temperature=config.generation_temperature,
            randomness=config.randomness,
            min_add_len=config.min_add_len,
            seed=config.seed,
            max_completion_length=config.max_completion_length,
            length_path=config.length_path,
        )
        _log("generating molecules")
        rollout = policy.rollout_specs(
            specs=specs,
            generation_batch_size=min(config.generation_batch_size, len(specs)),
            seed=config.seed,
        )
        _log("scoring molecules")
        records = reward_model.score(rollout.smiles)
    finally:
        reward_model.close()
        del policy

    if len(records) != config.num_samples:
        raise RuntimeError(f"Expected {config.num_samples} reward records, got {len(records)}")

    rows = []
    for sample_index, record in enumerate(records):
        rows.append(
            {
                "model_id": model.model_id,
                "row_id": model.row_id,
                "row_title": model.row_title,
                "slot_index": int(model.slot_index),
                "slot_label": model.slot_label,
                "display_name": model.display_name,
                "checkpoint_path": model.checkpoint_path,
                "sample_index": sample_index,
                "generation_temperature": float(config.generation_temperature),
                "randomness": float(config.randomness),
                "reward": float(record.reward),
                "is_valid": bool(record.is_valid),
                "alert_hit": bool(record.alert_hit),
                "qed": record.qed,
                "sa": record.sa,
                "sa_score": record.sa_score,
                "soft_reward": record.soft_reward,
                "smiles": record.smiles,
            }
        )
    return rows


def _write_rows(model_dir: Path, rows: list[dict]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    records_path = model_dir / "records.jsonl"
    summary_path = model_dir / "summary.json"
    with records_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    valid_rows = [row for row in rows if row["is_valid"] and row["smiles"] is not None]
    summary = {
        "num_rows": len(rows),
        "num_valid_rows": len(valid_rows),
        "valid_fraction": float(len(valid_rows) / len(rows)),
        "soft_reward_mean_all": float(sum(float(row["reward"]) for row in rows) / len(rows)),
        "soft_reward_mean_valid": (
            float(sum(float(row["soft_reward"]) for row in valid_rows) / len(valid_rows))
            if valid_rows
            else None
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    config = load_dynamics_config(args.config)
    model = _resolve_model(config, model_id=args.model_id, model_index=args.model_index)
    _log(f"collecting model_id={model.model_id}")
    rows = _build_rows(config, model)
    _write_rows(model_output_dir(config, model), rows)
    _log(f"wrote {len(rows)} rows for model_id={model.model_id}")


if __name__ == "__main__":
    main()
