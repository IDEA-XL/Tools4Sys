import sys
import types

import numpy as np

from genmol.mm import crossdocked

from genmol.mm.crossdocked import reconstruct_residue_pocket_from_entry


def test_reconstruct_residue_pocket_from_residue_level_entry():
    entry = {
        'protein_amino_acid': np.array([0, 5], dtype=np.int64),
        'protein_pos_N': np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        'protein_pos_CA': np.array([[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float32),
        'protein_pos_C': np.array([[1.0, 2.0, 0.0], [2.0, 2.0, 0.0]], dtype=np.float32),
    }
    sequence, coords = reconstruct_residue_pocket_from_entry(entry)
    assert sequence == 'AG'
    assert coords.shape == (2, 3, 3)


def test_reconstruct_residue_pocket_from_atom_level_entry():
    entry = {
        'protein_pos': np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.1, 0.0, 0.0],
                [1.2, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        'protein_atom_name': ['N', 'CA', 'C', 'N', 'CA', 'C'],
        'protein_atom_to_aa_type': np.array([0, 0, 0, 5, 5, 5], dtype=np.int64),
    }
    sequence, coords = reconstruct_residue_pocket_from_entry(entry)
    assert sequence == 'AG'
    assert coords.shape == (2, 3, 3)


def test_smiles_to_safe_uses_genmol_preprocess_settings(monkeypatch):
    calls = {}

    class DummyConverter:
        def __init__(self, *, ignore_stereo):
            calls['ignore_stereo'] = ignore_stereo

        def encoder(self, smiles, *, allow_empty):
            calls['smiles'] = smiles
            calls['allow_empty'] = allow_empty
            return 'ABC'

    fake_safe = types.SimpleNamespace(SAFEConverter=DummyConverter)
    monkeypatch.setitem(sys.modules, 'safe', fake_safe)

    result = crossdocked.smiles_to_safe('C/C=C\\O')
    assert result == 'ABC'
    assert calls == {
        'ignore_stereo': True,
        'smiles': 'C/C=C\\O',
        'allow_empty': True,
    }


def test_smiles_to_safe_relabels_fragmentation_errors(monkeypatch):
    class DummyConverter:
        def __init__(self, *, ignore_stereo):
            pass

        def encoder(self, smiles, *, allow_empty):
            raise RuntimeError('Slicing algorithms did not return any bonds that can be cut !')

    fake_safe = types.SimpleNamespace(SAFEConverter=DummyConverter)
    monkeypatch.setitem(sys.modules, 'safe', fake_safe)

    try:
        crossdocked.smiles_to_safe('c1ccccc1')
    except ValueError as exc:
        assert str(exc).startswith('SAFE conversion failed:')
    else:
        raise AssertionError('Expected SAFE conversion failure to be relabeled')
