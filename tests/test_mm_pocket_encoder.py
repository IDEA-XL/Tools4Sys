from types import SimpleNamespace

import pytest
import torch

from genmol.mm.pocket_encoder import ESMPocketEncoder, _patch_biotite_filter_backbone_compat


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


def test_encode_passes_target_device_to_coord_batch_converter():
    class FakeBatchConverter:
        def __init__(self):
            self.devices = []

        def __call__(self, batch, device=None):
            self.devices.append(device)
            coords = torch.zeros((1, 4, 3, 3), dtype=torch.float32, device=device)
            confidence = torch.ones((1, 4), dtype=torch.float32, device=device)
            tokens = torch.zeros((1, 4), dtype=torch.long, device=device)
            padding_mask = torch.zeros((1, 4), dtype=torch.bool, device=device)
            return coords, confidence, None, tokens, padding_mask

    class FakeEncoder:
        def forward(self, coords, padding_mask, confidence, return_all_hiddens=False):
            assert coords.device == confidence.device == padding_mask.device
            output = torch.zeros((6, 1, 8), dtype=torch.float32, device=coords.device)
            return {'encoder_out': [output]}

    encoder = ESMPocketEncoder.__new__(ESMPocketEncoder)
    encoder.device = torch.device('cpu')
    encoder._batch_converter = FakeBatchConverter()
    encoder.model = SimpleNamespace(encoder=FakeEncoder())

    outputs = encoder.encode([torch.zeros((2, 3, 3), dtype=torch.float32)])

    assert encoder._batch_converter.devices == [torch.device('cpu')]
    assert len(outputs) == 1
    assert outputs[0].shape == (4, 8)
    assert outputs[0].device.type == 'cpu'
