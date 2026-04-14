import pytest

from genmol.mm.checkpoint import (
    POCKET_PREFIX_MM_VARIANT,
    UNIMODAL_VARIANT,
    infer_checkpoint_variant,
    require_multimodal_checkpoint,
    require_unimodal_checkpoint,
    stamp_checkpoint_variant,
)


def test_infer_checkpoint_variant_defaults_to_unimodal():
    checkpoint = {}
    assert infer_checkpoint_variant(checkpoint) == UNIMODAL_VARIANT


def test_require_unimodal_rejects_multimodal():
    checkpoint = {}
    stamp_checkpoint_variant(checkpoint, POCKET_PREFIX_MM_VARIANT)
    with pytest.raises(ValueError, match='Expected a unimodal GenMol checkpoint'):
        require_unimodal_checkpoint(checkpoint, checkpoint_path='dummy.ckpt')


def test_require_multimodal_rejects_legacy_unimodal():
    checkpoint = {}
    with pytest.raises(ValueError, match='Checkpoint variant mismatch'):
        require_multimodal_checkpoint(checkpoint, checkpoint_path='dummy.ckpt')
