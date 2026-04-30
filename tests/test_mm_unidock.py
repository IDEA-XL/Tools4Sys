from pathlib import Path

import numpy as np

from genmol.mm.utils import unidock as unidock_module
from genmol.mm.utils.unidock import (
    UniDockConfig,
    UniDockScorer,
    _build_receptor_pdb_lines,
    _element_symbol_from_atomic_number,
    _normalize_receptor_atom_name,
)


def _write_fake_unidock_binary(path: Path):
    path.write_text(
        """#!/usr/bin/env python3
import pathlib
import sys

args = sys.argv[1:]
gpu_batch_start = args.index('--gpu_batch') + 1
gpu_batch_end = gpu_batch_start
while gpu_batch_end < len(args) and not args[gpu_batch_end].startswith('--'):
    gpu_batch_end += 1
ligands = [pathlib.Path(item) for item in args[gpu_batch_start:gpu_batch_end]]
output_dir = pathlib.Path(args[args.index('--dir') + 1])
output_dir.mkdir(parents=True, exist_ok=True)

for idx, ligand in enumerate(ligands):
    output_path = output_dir / f"{ligand.stem}_out.sdf"
    output_path.write_text(
        f"> <Uni-Dock RESULT>\\nENERGY=-{idx + 1}.500  LOWER_BOUND=0.0  UPPER_BOUND=0.0\\n\\n$$$$\\n"
    )
"""
    )
    path.chmod(0o755)


class _FakeCrossDockedRawEntryStore:
    def __init__(self, _lmdb_path: str):
        self._entries = {
            1: {
                'protein_pos': np.asarray(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
                'protein_atom_name': ['N', 'CA', 'C'],
                'protein_atom_to_aa_type': np.asarray([0, 0, 0], dtype=np.int64),
                'protein_element': np.asarray([7, 6, 6], dtype=np.int64),
                'ligand_pos': np.asarray(
                    [
                        [0.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
                'ligand_element': np.asarray([6, 6], dtype=np.int64),
            },
            2: {
                'protein_pos': np.asarray(
                    [
                        [0.0, 1.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [2.0, 1.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
                'protein_atom_name': ['N', 'CA', 'C'],
                'protein_atom_to_aa_type': np.asarray([1, 1, 1], dtype=np.int64),
                'protein_element': np.asarray([7, 6, 6], dtype=np.int64),
                'ligand_pos': np.asarray(
                    [
                        [1.0, 2.0, 3.0],
                        [3.0, 2.0, 3.0],
                    ],
                    dtype=np.float32,
                ),
                'ligand_element': np.asarray([6, 6], dtype=np.int64),
            },
        }

    def close(self):
        return None

    def get_entry(self, pocket_entry: dict) -> dict:
        return self._entries[int(pocket_entry['source_index'])]


def test_unidock_scorer_deduplicates_smiles_within_pocket(tmp_path, monkeypatch):
    lmdb_path = tmp_path / 'fake.lmdb'
    lmdb_path.write_text('placeholder')
    fake_binary = tmp_path / 'fake_unidock.py'
    _write_fake_unidock_binary(fake_binary)

    def fake_prepare_ligand_sdf(smiles: str, center, output_path: Path):
        output_path.write_text(f'{smiles} @ {list(center)}\n')

    monkeypatch.setattr(unidock_module, '_CrossDockedRawEntryStore', _FakeCrossDockedRawEntryStore)
    monkeypatch.setattr(unidock_module, '_prepare_ligand_sdf', fake_prepare_ligand_sdf)
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '8')
    monkeypatch.setenv('LOCAL_WORLD_SIZE', '2')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')

    scorer = UniDockScorer(
        UniDockConfig(
            binary_path=str(fake_binary),
            crossdocked_lmdb_path=str(lmdb_path),
            device='cuda:0',
            batch_size=2,
            score_cache_size=16,
        )
    )

    smiles_list = ['CCO', 'CCO', 'CCC', 'CCN']
    pocket_entries = [
        {'source_index': 1, 'pocket_coords': [[[0.0, 0.0, 0.0]]]},
        {'source_index': 1, 'pocket_coords': [[[0.0, 0.0, 0.0]]]},
        {'source_index': 1, 'pocket_coords': [[[0.0, 0.0, 0.0]]]},
        {'source_index': 2, 'pocket_coords': [[[1.0, 2.0, 3.0]]]},
    ]

    try:
        scores = scorer.score(smiles_list, pocket_entries)
    finally:
        scorer.close()

    assert scores == [-1.5, -1.5, -2.5, -1.5]
    assert scorer.last_score_stats['unidock_unique_pocket_count'] == 2
    assert scorer.last_score_stats['unidock_unique_smiles_count'] == 3
    assert scorer.last_score_stats['unidock_chunk_count'] == 2
    assert scorer.last_score_stats['unidock_prepare_worker_count'] == 4
    assert scorer.last_score_stats['unidock_score_success_count'] == 4
    assert scorer.last_score_stats['unidock_score_failure_count'] == 0


def test_unidock_receptor_normalizes_deuterium_names():
    assert _normalize_receptor_atom_name('D', 1) == 'H'
    assert _normalize_receptor_atom_name('DD21', 1) == 'HD21'
    assert _normalize_receptor_atom_name('DH', 1) == 'HH'
    assert _normalize_receptor_atom_name('HA2', 1) == 'HA2'


def test_unidock_receptor_uses_atomic_number_for_element_symbol():
    assert _element_symbol_from_atomic_number(1) == 'H'
    assert _element_symbol_from_atomic_number(6) == 'C'
    assert _element_symbol_from_atomic_number(7) == 'N'
    assert _element_symbol_from_atomic_number(8) == 'O'


def test_unidock_receptor_build_lines_rewrite_deuterium_to_hydrogen():
    raw_entry = {
        'protein_pos': np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        'protein_atom_name': ['N', 'CA', 'C', 'O', 'D', 'DD21', 'DH'],
        'protein_atom_to_aa_type': np.asarray([5, 5, 5, 5, 5, 5, 5], dtype=np.int64),
        'protein_element': np.asarray([7, 6, 6, 8, 1, 1, 1], dtype=np.int64),
    }

    lines = _build_receptor_pdb_lines(raw_entry)
    line_blob = ''.join(lines)
    assert ' D ' not in line_blob
    assert ' DD21 ' not in line_blob
    assert ' DH ' not in line_blob
    assert ' H ' in line_blob
    assert ' HD21 ' in line_blob
    assert ' HH ' in line_blob
    assert line_blob.count('           H') == 3
