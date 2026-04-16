import importlib
import sys
import types

import pytest

from progen2.rewards.foldability import _require_openfold_attention_core_extension


def test_require_openfold_attention_core_extension_accepts_importable_module(monkeypatch):
    fake_module = types.ModuleType('attn_core_inplace_cuda')
    monkeypatch.setitem(sys.modules, 'attn_core_inplace_cuda', fake_module)
    _require_openfold_attention_core_extension()


def test_require_openfold_attention_core_extension_raises_when_missing(monkeypatch):
    monkeypatch.delitem(sys.modules, 'attn_core_inplace_cuda', raising=False)

    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == 'attn_core_inplace_cuda':
            raise ImportError('missing test extension')
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, 'import_module', fake_import_module)

    with pytest.raises(RuntimeError, match='compiled OpenFold attention extension'):
        _require_openfold_attention_core_extension()
