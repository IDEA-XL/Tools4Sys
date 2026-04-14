import os

import torch


CHECKPOINT_VARIANT_KEY = 'genmol_model_variant'
UNIMODAL_VARIANT = 'genmol_unimodal'
POCKET_PREFIX_MM_VARIANT = 'pocket_prefix_mm'


def infer_checkpoint_variant(checkpoint):
    variant = checkpoint.get(CHECKPOINT_VARIANT_KEY)
    if variant is None:
        return UNIMODAL_VARIANT
    return str(variant)


def stamp_checkpoint_variant(checkpoint, variant):
    checkpoint[CHECKPOINT_VARIANT_KEY] = str(variant)


def load_checkpoint_payload(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'checkpoint not found: {checkpoint_path}')
    return torch.load(checkpoint_path, map_location='cpu', weights_only=False)


def require_checkpoint_variant(checkpoint, expected_variant, checkpoint_path):
    variant = infer_checkpoint_variant(checkpoint)
    if variant != expected_variant:
        raise ValueError(
            'Checkpoint variant mismatch: '
            f'expected {expected_variant!r}, got {variant!r} for {checkpoint_path}'
        )


def require_unimodal_checkpoint(checkpoint, checkpoint_path):
    variant = infer_checkpoint_variant(checkpoint)
    if variant != UNIMODAL_VARIANT:
        raise ValueError(
            'Expected a unimodal GenMol checkpoint, got '
            f'{variant!r} for {checkpoint_path}'
        )


def require_multimodal_checkpoint(checkpoint, checkpoint_path):
    require_checkpoint_variant(
        checkpoint=checkpoint,
        expected_variant=POCKET_PREFIX_MM_VARIANT,
        checkpoint_path=checkpoint_path,
    )
