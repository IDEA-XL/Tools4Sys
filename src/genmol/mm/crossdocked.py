import json
import os
import pickle
from dataclasses import dataclass

import numpy as np
import torch

from genmol.utils.utils_data import get_tokenizer


_AA_INDEX_TO_SYMBOL = {
    0: 'A',
    1: 'C',
    2: 'D',
    3: 'E',
    4: 'F',
    5: 'G',
    6: 'H',
    7: 'I',
    8: 'K',
    9: 'L',
    10: 'M',
    11: 'N',
    12: 'P',
    13: 'Q',
    14: 'R',
    15: 'S',
    16: 'T',
    17: 'V',
    18: 'W',
    19: 'Y',
}


@dataclass(frozen=True)
class ManifestBuildStats:
    total_samples_scanned: int
    valid_samples: int
    dropped_unassigned_split_samples: int
    dropped_overlength_samples: int
    dropped_malformed_pocket_samples: int
    dropped_safe_conversion_samples: int


def _normalize_array(value, dtype, label):
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        raise ValueError(f'{label} must be at least 1D')
    return array


def _decode_atom_name(atom_name):
    if isinstance(atom_name, bytes):
        return atom_name.decode('utf-8').strip()
    return str(atom_name).strip()


def reconstruct_residue_pocket_from_entry(entry):
    residue_keys = {'protein_amino_acid', 'protein_pos_N', 'protein_pos_CA', 'protein_pos_C'}
    if residue_keys.issubset(entry.keys()):
        amino_acid = _normalize_array(entry['protein_amino_acid'], np.int64, 'protein_amino_acid')
        pos_n = _normalize_array(entry['protein_pos_N'], np.float32, 'protein_pos_N')
        pos_ca = _normalize_array(entry['protein_pos_CA'], np.float32, 'protein_pos_CA')
        pos_c = _normalize_array(entry['protein_pos_C'], np.float32, 'protein_pos_C')
        if pos_n.shape != pos_ca.shape or pos_n.shape != pos_c.shape:
            raise ValueError('protein_pos_N/protein_pos_CA/protein_pos_C shapes must match')
        if pos_n.ndim != 2 or pos_n.shape[1] != 3:
            raise ValueError(
                'protein_pos_* arrays must have shape [num_residues, 3], '
                f'got {list(pos_n.shape)}'
            )
        if amino_acid.shape[0] != pos_n.shape[0]:
            raise ValueError(
                'protein_amino_acid length must match protein_pos_* residue count: '
                f'{amino_acid.shape[0]} vs {pos_n.shape[0]}'
            )
        sequence = ''.join(_AA_INDEX_TO_SYMBOL.get(int(item), 'X') for item in amino_acid.tolist())
        if 'X' in sequence:
            raise ValueError('protein_amino_acid contains unsupported residue indices')
        coords = np.stack([pos_n, pos_ca, pos_c], axis=1)
        return sequence, coords

    atom_keys = {'protein_pos', 'protein_atom_name', 'protein_atom_to_aa_type'}
    if not atom_keys.issubset(entry.keys()):
        missing = sorted(atom_keys.difference(entry.keys()))
        raise ValueError(f'Missing required atom-level protein keys: {missing}')

    atom_positions = _normalize_array(entry['protein_pos'], np.float32, 'protein_pos')
    atom_names = entry['protein_atom_name']
    atom_to_aa_type = _normalize_array(entry['protein_atom_to_aa_type'], np.int64, 'protein_atom_to_aa_type')
    if atom_positions.ndim != 2 or atom_positions.shape[1] != 3:
        raise ValueError(f'protein_pos must have shape [num_atoms, 3], got {list(atom_positions.shape)}')
    if len(atom_names) != atom_positions.shape[0] or atom_to_aa_type.shape[0] != atom_positions.shape[0]:
        raise ValueError('protein atom-level arrays must share the same length')

    decoded_atom_names = [_decode_atom_name(item) for item in atom_names]
    residue_start_indices = [idx for idx, name in enumerate(decoded_atom_names) if name == 'N']
    if not residue_start_indices:
        raise ValueError('Failed to reconstruct residues: no backbone N atoms found')

    sequence = []
    residue_coords = []
    residue_start_indices.append(len(decoded_atom_names))
    for start_idx, end_idx in zip(residue_start_indices[:-1], residue_start_indices[1:]):
        residue_atom_names = decoded_atom_names[start_idx:end_idx]
        residue_atom_types = atom_to_aa_type[start_idx:end_idx]
        residue_positions = atom_positions[start_idx:end_idx]
        unique_types = np.unique(residue_atom_types)
        if unique_types.size != 1:
            raise ValueError(
                'Failed to reconstruct residue boundaries deterministically: '
                f'multiple amino acid types in a residue segment {unique_types.tolist()}'
            )

        aa_index = int(unique_types[0])
        aa_symbol = _AA_INDEX_TO_SYMBOL.get(aa_index)
        if aa_symbol is None:
            raise ValueError(f'Unsupported amino acid index: {aa_index}')

        atom_name_to_pos = {}
        for atom_name, atom_pos in zip(residue_atom_names, residue_positions):
            atom_name_to_pos.setdefault(atom_name, atom_pos)
        missing_backbone = [name for name in ('N', 'CA', 'C') if name not in atom_name_to_pos]
        if missing_backbone:
            raise ValueError(
                'Failed to reconstruct residue backbone coordinates: '
                f'missing {missing_backbone} in residue with amino acid index {aa_index}'
            )

        sequence.append(aa_symbol)
        residue_coords.append(
            np.stack(
                [
                    atom_name_to_pos['N'],
                    atom_name_to_pos['CA'],
                    atom_name_to_pos['C'],
                ],
                axis=0,
            )
        )

    if not residue_coords:
        raise ValueError('Failed to reconstruct any residue coordinates')
    return ''.join(sequence), np.stack(residue_coords, axis=0).astype(np.float32)


def smiles_to_safe(smiles):
    try:
        import safe as sf
    except ImportError as exc:
        raise ImportError(
            'safe-mol is required to build the CrossDocked multimodal manifest'
        ) from exc

    converter = sf.SAFEConverter(slicer=None)
    return converter.encoder(smiles, allow_empty=False)


def build_manifest_entry(entry, source_index, split, max_total_positions):
    if 'ligand_smiles' not in entry:
        raise ValueError('Missing ligand_smiles in CrossDocked entry')

    ligand_smiles = entry['ligand_smiles']
    if isinstance(ligand_smiles, bytes):
        ligand_smiles = ligand_smiles.decode('utf-8')
    ligand_smiles = str(ligand_smiles).strip()
    if not ligand_smiles:
        raise ValueError('ligand_smiles is empty')

    safe_string = smiles_to_safe(ligand_smiles)
    tokenizer = get_tokenizer()
    ligand_token_length = len(tokenizer(safe_string)['input_ids'])
    sequence, coords = reconstruct_residue_pocket_from_entry(entry)
    residue_count = int(coords.shape[0])
    total_length = residue_count + ligand_token_length
    if total_length > max_total_positions:
        raise ValueError(
            'Sample exceeds max_total_positions: '
            f'residue_count={residue_count} ligand_token_length={ligand_token_length} '
            f'total_length={total_length} max_total_positions={max_total_positions}'
        )

    return {
        'source_index': int(source_index),
        'split': str(split),
        'safe': safe_string,
        'ligand_smiles': ligand_smiles,
        'ligand_token_length': int(ligand_token_length),
        'pocket_sequence': sequence,
        'pocket_coords': coords.astype(np.float32),
        'residue_count': residue_count,
        'total_length': int(total_length),
        'protein_filename': entry.get('protein_filename'),
        'ligand_filename': entry.get('ligand_filename'),
    }


def load_crossdocked_split_map(split_path):
    if not os.path.exists(split_path):
        raise FileNotFoundError(f'split file not found: {split_path}')
    split_payload = torch.load(split_path, map_location='cpu', weights_only=False)
    if not isinstance(split_payload, dict):
        raise ValueError(f'Expected dict split payload in {split_path}')

    index_to_split = {}
    for split_name, indices in split_payload.items():
        if split_name not in {'train', 'val', 'test'}:
            raise ValueError(f'Unsupported split name in split file: {split_name}')
        for index in indices:
            index_int = int(index)
            if index_int in index_to_split:
                raise ValueError(f'Duplicate split assignment for index {index_int}')
            index_to_split[index_int] = split_name
    return index_to_split


def _open_lmdb(lmdb_path):
    try:
        import lmdb
    except ImportError as exc:
        raise ImportError('lmdb is required to read CrossDocked processed_final.lmdb') from exc

    if not os.path.exists(lmdb_path):
        raise FileNotFoundError(f'CrossDocked LMDB not found: {lmdb_path}')
    return lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=os.path.isdir(lmdb_path),
        max_readers=1,
    )


def iter_crossdocked_entries(lmdb_path):
    env = _open_lmdb(lmdb_path)
    try:
        with env.begin(write=False) as txn:
            num_examples_raw = txn.get(b'num_examples')
            if num_examples_raw is not None:
                num_examples = int(num_examples_raw.decode('utf-8'))
                for index in range(num_examples):
                    payload = txn.get(str(index).encode('utf-8'))
                    if payload is None:
                        raise ValueError(f'CrossDocked LMDB is missing entry index {index}')
                    yield index, pickle.loads(payload)
                return

            cursor = txn.cursor()
            found_numeric_key = False
            for raw_key, payload in cursor:
                try:
                    decoded_key = raw_key.decode('utf-8')
                except UnicodeDecodeError as exc:
                    raise ValueError(
                        'CrossDocked LMDB is missing num_examples and contains a non-UTF8 key: '
                        f'{raw_key!r}'
                    ) from exc
                if not decoded_key.isdigit():
                    raise ValueError(
                        'CrossDocked LMDB is missing num_examples and contains a non-numeric key: '
                        f'{decoded_key!r}'
                    )
                found_numeric_key = True
                yield int(decoded_key), pickle.loads(payload)
            if not found_numeric_key:
                raise ValueError(
                    'CrossDocked LMDB contains no numeric sample keys and no num_examples metadata: '
                    f'{lmdb_path}'
                )
    finally:
        env.close()


def build_crossdocked_manifest(lmdb_path, split_path, max_total_positions):
    split_map = load_crossdocked_split_map(split_path)
    entries = []
    total_samples_scanned = 0
    dropped_unassigned_split_samples = 0
    dropped_overlength_samples = 0
    dropped_malformed_pocket_samples = 0
    dropped_safe_conversion_samples = 0

    for source_index, raw_entry in iter_crossdocked_entries(lmdb_path):
        total_samples_scanned += 1
        split = split_map.get(source_index)
        if split is None:
            dropped_unassigned_split_samples += 1
            continue
        try:
            manifest_entry = build_manifest_entry(
                entry=raw_entry,
                source_index=source_index,
                split=split,
                max_total_positions=max_total_positions,
            )
        except ImportError:
            raise
        except Exception as exc:
            message = str(exc)
            if 'safe' in message.lower():
                dropped_safe_conversion_samples += 1
                continue
            if 'max_total_positions' in message:
                dropped_overlength_samples += 1
                continue
            dropped_malformed_pocket_samples += 1
            continue
        entries.append(manifest_entry)

    stats = ManifestBuildStats(
        total_samples_scanned=total_samples_scanned,
        valid_samples=len(entries),
        dropped_unassigned_split_samples=dropped_unassigned_split_samples,
        dropped_overlength_samples=dropped_overlength_samples,
        dropped_malformed_pocket_samples=dropped_malformed_pocket_samples,
        dropped_safe_conversion_samples=dropped_safe_conversion_samples,
    )
    return entries, stats


def save_crossdocked_manifest(entries, stats, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        'entries': entries,
        'stats': stats.__dict__,
    }
    torch.save(payload, output_path)
    stats_path = f'{output_path}.stats.json'
    with open(stats_path, 'w') as handle:
        json.dump(stats.__dict__, handle, sort_keys=True, indent=2)
    return stats_path


def load_crossdocked_manifest(manifest_path, split):
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f'manifest not found: {manifest_path}')
    payload = torch.load(manifest_path, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f'Expected dict payload in manifest: {manifest_path}')
    if 'entries' not in payload or 'stats' not in payload:
        raise ValueError(f'Manifest payload missing entries/stats: {manifest_path}')
    entries = payload['entries']
    if not isinstance(entries, list):
        raise ValueError(f'Manifest entries must be a list: {manifest_path}')
    filtered = [entry for entry in entries if entry['split'] == split]
    if not filtered:
        raise ValueError(f'No entries found for split {split!r} in manifest {manifest_path}')
    return filtered, payload['stats']
