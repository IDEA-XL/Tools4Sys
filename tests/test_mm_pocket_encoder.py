from types import SimpleNamespace

import pytest

from genmol.mm.pocket_encoder import _patch_biotite_filter_backbone_compat


def test_patch_biotite_filter_backbone_compat_noop_if_present():
    sentinel = object()
    module = SimpleNamespace(filter_backbone=sentinel)
    changed = _patch_biotite_filter_backbone_compat(module)
    assert changed is False
    assert module.filter_backbone is sentinel


def test_patch_biotite_filter_backbone_compat_aliases_filter_peptide_backbone():
    sentinel = object()
    module = SimpleNamespace(filter_peptide_backbone=sentinel)
    changed = _patch_biotite_filter_backbone_compat(module)
    assert changed is True
    assert module.filter_backbone is sentinel


def test_patch_biotite_filter_backbone_compat_fail_fast_without_compatible_api():
    module = SimpleNamespace(check_backbone_continuity=object())
    with pytest.raises(ImportError, match='filter_backbone'):
        _patch_biotite_filter_backbone_compat(module)
