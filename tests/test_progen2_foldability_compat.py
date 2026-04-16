import sys
import types

from progen2.rewards.foldability import _ensure_openfold_deepspeed_compat


def test_ensure_openfold_deepspeed_compat_shims_missing_utils_api(monkeypatch):
    fake_deepspeed = types.SimpleNamespace(
        utils=types.SimpleNamespace(),
        comm=types.SimpleNamespace(is_initialized=lambda: False),
    )
    monkeypatch.setitem(sys.modules, 'deepspeed', fake_deepspeed)

    _ensure_openfold_deepspeed_compat()

    assert fake_deepspeed.utils.is_initialized is fake_deepspeed.comm.is_initialized
