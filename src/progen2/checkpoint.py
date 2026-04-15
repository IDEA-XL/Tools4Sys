import os

import torch


CHECKPOINT_VARIANT_KEY = 'protein_rl_variant'
PROGEN2_SGRPO_VARIANT = 'progen2_sgrpo'


def stamp_checkpoint_variant(checkpoint, variant):
    checkpoint[CHECKPOINT_VARIANT_KEY] = str(variant)


def infer_checkpoint_variant(checkpoint):
    return str(checkpoint.get(CHECKPOINT_VARIANT_KEY, PROGEN2_SGRPO_VARIANT))


def require_checkpoint_variant(checkpoint, expected_variant, checkpoint_path):
    variant = infer_checkpoint_variant(checkpoint)
    if variant != expected_variant:
        raise ValueError(
            'Checkpoint variant mismatch: '
            f'expected {expected_variant!r}, got {variant!r} for {checkpoint_path}'
        )


def load_checkpoint_payload(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'checkpoint not found: {checkpoint_path}')
    return torch.load(checkpoint_path, map_location='cpu', weights_only=False)
