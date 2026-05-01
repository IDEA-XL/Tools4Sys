from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Lipinski

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.realpath('.'), 'src'))

from genmol.mm.docking import _embed_ligand_from_smiles, _translate_ligand_to_center, _write_ligand_sdf
from genmol.mm.trainer import PocketPrefixCpGRPOTrainer, load_config


logger = logging.getLogger(__name__)


class _ReplayStop(RuntimeError):
    pass


def _stable_key(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def _configure_logging(level_name: str):
    log_level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )


def _parse_expected_failure_command(log_path: str, target_source_index: int) -> list[tuple[str, str]]:
    text = Path(log_path).read_text(errors='replace')
    pattern = re.compile(
        rf'command=.*?/source_{int(target_source_index)}\.pdb(?P<cmd>.*?)(?:\n\[rank|\nreturncode=1)',
        re.S,
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError(
            f'Failed to find a Uni-Dock failing command for source_{target_source_index} in {log_path}'
        )
    ligand_pairs = re.findall(r'lig_(\d{4})_([0-9a-f]{16})\.sdf', match.group('cmd'))
    if not ligand_pairs:
        raise ValueError(f'Failed to parse ligand hashes from failing command in {log_path}')
    return ligand_pairs


def _canonical_props(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        raise ValueError(f'RDKit failed to parse replay SMILES: {smiles!r}')
    return {
        'smiles': smiles,
        'stable_key': _stable_key(smiles),
        'canonical_smiles': Chem.MolToSmiles(mol),
        'num_atoms': int(mol.GetNumAtoms()),
        'num_heavy_atoms': int(mol.GetNumHeavyAtoms()),
        'num_fragments': int(len(Chem.GetMolFrags(mol))),
        'formal_charge': int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
        'num_rotatable_bonds': int(Lipinski.CalcNumRotatableBonds(mol)),
        'atomic_numbers': [int(atom.GetAtomicNum()) for atom in mol.GetAtoms()],
    }


def _prepare_sdf_analysis(smiles: str, center: np.ndarray, workdir: Path) -> dict:
    ligand_path = workdir / f'lig_{_stable_key(smiles)}.sdf'
    mol = _embed_ligand_from_smiles(smiles)
    translated = _translate_ligand_to_center(mol, center)
    _write_ligand_sdf(translated, ligand_path)
    sdf_text = ligand_path.read_text()
    supplier = Chem.SDMolSupplier(str(ligand_path), removeHs=False)
    molecules = [item for item in supplier if item is not None]
    return {
        'sdf_path': str(ligand_path),
        'sdf_size_bytes': int(ligand_path.stat().st_size),
        'sdf_head': sdf_text.splitlines()[:12],
        'reloaded_molecule_count': int(len(molecules)),
        'reloaded_num_atoms': None if not molecules else int(molecules[0].GetNumAtoms()),
        'reloaded_num_heavy_atoms': None if not molecules else int(molecules[0].GetNumHeavyAtoms()),
    }


def _run_unidock_subset(scorer, receptor_path: Path, pocket_center: np.ndarray, smiles_subset: list[str], pocket_key: str):
    return scorer._run_unidock_chunk(
        receptor_path=receptor_path,
        pocket_center=pocket_center,
        smiles_chunk=smiles_subset,
        pocket_key=pocket_key,
    )


def _find_minimal_failing_subset(scorer, receptor_path: Path, pocket_center: np.ndarray, smiles_list: list[str], pocket_key: str):
    def fails(subset: list[str]) -> tuple[bool, str | None]:
        try:
            _run_unidock_subset(scorer, receptor_path, pocket_center, subset, pocket_key)
            return False, None
        except Exception as exc:  # noqa: BLE001
            return True, f'{type(exc).__name__}: {exc}'

    full_fails, full_error = fails(smiles_list)
    if not full_fails:
        raise ValueError('Expected full Uni-Dock subset to fail during replay isolation, but it succeeded')

    current = list(smiles_list)
    current_error = full_error
    while len(current) > 1:
        midpoint = len(current) // 2
        left = current[:midpoint]
        right = current[midpoint:]
        left_fails, left_error = fails(left)
        if left_fails:
            current = left
            current_error = left_error
            continue
        right_fails, right_error = fails(right)
        if right_fails:
            current = right
            current_error = right_error
            continue
        break

    singles = []
    for smiles in current:
        single_fails, single_error = fails([smiles])
        singles.append(
            {
                'smiles': smiles,
                'stable_key': _stable_key(smiles),
                'fails_individually': bool(single_fails),
                'error': single_error,
            }
        )
    return {
        'full_subset_size': int(len(smiles_list)),
        'minimal_subset_size': int(len(current)),
        'minimal_subset_hashes': [_stable_key(smiles) for smiles in current],
        'minimal_subset_smiles': current,
        'minimal_subset_error': current_error,
        'single_ligand_results': singles,
    }


def _write_report(path: str, payload: dict):
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume_from_checkpoint', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--target_source_index', type=int, default=None)
    parser.add_argument('--expected_failure_log', default=None)
    parser.add_argument('--report_path', required=True)
    parser.add_argument(
        '--mode',
        required=True,
        choices=('capture_no_atoms', 'isolate_internal_error', 'catch_first_no_atoms'),
    )
    parser.add_argument('--target_ligand_hash', default=None)
    parser.add_argument('--max_steps', required=True, type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    _configure_logging(config.log_level)
    config.max_steps = int(args.max_steps)
    config.save_steps = max(config.save_steps, config.max_steps + 1000)
    config.report_to = []
    config.log_completions = False
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'catch_first_no_atoms':
        expected_hashes_in_order = []
        expected_hash_set = set()
    else:
        if args.target_source_index is None:
            raise ValueError('--target_source_index is required unless mode=catch_first_no_atoms')
        if not args.expected_failure_log:
            raise ValueError('--expected_failure_log is required unless mode=catch_first_no_atoms')
        expected_pairs = _parse_expected_failure_command(args.expected_failure_log, args.target_source_index)
        expected_hashes_in_order = [hash_value for _index, hash_value in expected_pairs]
        expected_hash_set = set(expected_hashes_in_order)

    trainer = PocketPrefixCpGRPOTrainer(config=config, output_dir=args.output_dir)
    scorer = trainer.reward_model._unidock
    if scorer is None:
        raise ValueError('Replay requires Uni-Dock reward to be enabled')
    raw_entry_store = scorer._raw_entry_store

    original_score = scorer.score
    original_run_unidock_chunk = scorer._run_unidock_chunk
    captured = False
    seen_target_batches: list[dict] = []

    def debug_run_unidock_chunk(*, receptor_path, pocket_center, smiles_chunk, pocket_key):
        nonlocal captured
        try:
            return original_run_unidock_chunk(
                receptor_path=receptor_path,
                pocket_center=pocket_center,
                smiles_chunk=smiles_chunk,
                pocket_key=pocket_key,
            )
        except Exception as exc:  # noqa: BLE001
            if args.mode != 'catch_first_no_atoms' or 'No atoms in this ligand.' not in str(exc):
                raise
            match = re.search(r'lig_(\d{4})_([0-9a-f]{16})\.sdf', str(exc))
            if match is None:
                raise ValueError(
                    'Caught Uni-Dock no-atoms failure but failed to parse ligand hash from exception'
                ) from exc
            ligand_hash = match.group(2)
            stable_to_smiles = {_stable_key(smiles): smiles for smiles in smiles_chunk}
            if ligand_hash not in stable_to_smiles:
                raise ValueError(
                    f'Caught Uni-Dock no-atoms failure for hash {ligand_hash}, but hash not present in current chunk'
                ) from exc
            failing_smiles = stable_to_smiles[ligand_hash]
            payload = {
                'status': 'captured',
                'mode': args.mode,
                'global_step_before_step_increment': int(trainer.global_step),
                'generation_cycle_idx': int(trainer.generation_cycle_idx),
                'process_index': int(trainer.accelerator.process_index),
                'receptor_path': str(receptor_path),
                'pocket_key': pocket_key,
                'target_ligand_hash': ligand_hash,
                'target_ligand_properties': _canonical_props(failing_smiles),
                'chunk_size': int(len(smiles_chunk)),
                'chunk_hashes_in_order': [_stable_key(smiles) for smiles in smiles_chunk],
            }
            with tempfile.TemporaryDirectory(prefix='mm_unidock_no_atoms_') as temp_dir_str:
                temp_dir = Path(temp_dir_str)
                payload['target_ligand_sdf_analysis'] = _prepare_sdf_analysis(
                    failing_smiles,
                    pocket_center,
                    temp_dir,
                )
            captured = True
            _write_report(args.report_path, payload)
            raise _ReplayStop(f'Debug payload written to {args.report_path}') from exc

    def debug_score(smiles_list, pocket_entries):
        nonlocal captured
        if args.mode == 'catch_first_no_atoms':
            return original_score(smiles_list, pocket_entries)
        grouped = {}
        for smiles, pocket_entry in zip(smiles_list, pocket_entries):
            if 'source_index' not in pocket_entry:
                raise ValueError('Replay requires source_index in pocket_entry')
            grouped.setdefault(int(pocket_entry['source_index']), []).append((smiles, pocket_entry))

        current_source_entries = grouped.get(int(args.target_source_index))
        if current_source_entries is None:
            return original_score(smiles_list, pocket_entries)

        current_smiles = [smiles for smiles, _entry in current_source_entries]
        current_hashes = [_stable_key(smiles) for smiles in current_smiles]
        seen_target_batches.append(
            {
                'source_index': int(args.target_source_index),
                'hash_count': int(len(current_hashes)),
                'hashes_in_order': current_hashes,
                'expected_hash_count': int(len(expected_hashes_in_order)),
                'matches_expected_order': bool(current_hashes == expected_hashes_in_order),
                'matches_expected_set': bool(set(current_hashes) == expected_hash_set),
                'contains_target_ligand_hash': (
                    None if args.target_ligand_hash is None else bool(args.target_ligand_hash in current_hashes)
                ),
            }
        )

        if args.mode == 'capture_no_atoms':
            should_capture = args.target_ligand_hash is not None and args.target_ligand_hash in current_hashes
        else:
            should_capture = len(current_hashes) == len(expected_hashes_in_order) and set(current_hashes) == expected_hash_set

        if not should_capture:
            return original_score(smiles_list, pocket_entries)

        pocket_entry = current_source_entries[0][1]
        raw_entry = raw_entry_store.get_entry(pocket_entry)
        receptor_path = scorer._resolve_receptor_pdb_path(pocket_entry)
        pocket_center = scorer._resolve_native_ligand_center(raw_entry) if hasattr(scorer, '_resolve_native_ligand_center') else None
        if pocket_center is None:
            from genmol.mm.utils.unidock import _resolve_native_ligand_center

            pocket_center = _resolve_native_ligand_center(raw_entry)

        payload = {
            'global_step_before_step_increment': int(trainer.global_step),
            'generation_cycle_idx': int(trainer.generation_cycle_idx),
            'process_index': int(trainer.accelerator.process_index),
            'target_source_index': int(args.target_source_index),
            'expected_hashes_in_order': expected_hashes_in_order,
            'matched_hashes_in_order': current_hashes,
            'matched_smiles': current_smiles,
            'receptor_path': str(receptor_path),
            'protein_filename': raw_entry.get('protein_filename'),
            'ligand_filename': raw_entry.get('ligand_filename'),
            'seen_target_batches': seen_target_batches,
        }

        with tempfile.TemporaryDirectory(prefix='mm_unidock_replay_') as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            if args.mode == 'capture_no_atoms':
                if args.target_ligand_hash is None:
                    raise ValueError('--target_ligand_hash is required in capture_no_atoms mode')
                match_index = current_hashes.index(args.target_ligand_hash)
                failing_smiles = current_smiles[match_index]
                payload['target_ligand_hash'] = args.target_ligand_hash
                payload['target_ligand_index_in_chunk'] = int(match_index)
                payload['target_ligand_properties'] = _canonical_props(failing_smiles)
                payload['target_ligand_sdf_analysis'] = _prepare_sdf_analysis(
                    failing_smiles,
                    pocket_center,
                    temp_dir,
                )
            else:
                payload['batch_properties'] = [_canonical_props(smiles) for smiles in current_smiles]
                payload['minimal_failing_subset'] = _find_minimal_failing_subset(
                    scorer,
                    receptor_path,
                    pocket_center,
                    current_smiles,
                    scorer._pocket_cache_key(pocket_entry),
                )

        captured = True
        _write_report(args.report_path, payload)
        raise _ReplayStop(f'Debug payload written to {args.report_path}')

    scorer.score = debug_score
    scorer._run_unidock_chunk = debug_run_unidock_chunk

    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    except _ReplayStop:
        logger.info('Replay captured target batch and stopped intentionally')
    else:
        if not captured:
            _write_report(
                args.report_path,
                {
                    'status': 'no_match',
                    'mode': args.mode,
                    'target_source_index': int(args.target_source_index),
                    'target_ligand_hash': args.target_ligand_hash,
                    'expected_hashes_in_order': expected_hashes_in_order,
                    'seen_target_batches': seen_target_batches,
                    'global_step_after_run': int(trainer.global_step),
                    'generation_cycle_idx_after_run': int(trainer.generation_cycle_idx),
                },
            )
    finally:
        scorer.score = original_score
        scorer._run_unidock_chunk = original_run_unidock_chunk
        trainer.close()


if __name__ == '__main__':
    main()
