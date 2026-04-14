import numpy as np

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
