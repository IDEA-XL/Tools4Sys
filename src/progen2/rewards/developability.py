from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
from pathlib import Path

from progen2.rewards.common import iter_chunks, validate_batch_size
from progen2.rewards.liability import liability_reward


def _resolve_proteinsol_root(model_name_or_path):
    root = Path(model_name_or_path).expanduser().resolve()
    candidates = [root, root / 'protein-sol-sequence-prediction-software']
    for candidate in candidates:
        if (candidate / 'multiple_prediction_wrapper_export.sh').is_file():
            return candidate
    raise ValueError(
        'Protein-Sol model_name_or_path must point to the extracted official Protein-Sol software root '
        f'or its parent directory; missing multiple_prediction_wrapper_export.sh under {root}'
    )


def _normalize_sequence_id(raw_sequence_id):
    sequence_id = raw_sequence_id.strip()
    if sequence_id.startswith('>'):
        sequence_id = sequence_id[1:].strip()
    if not sequence_id:
        raise ValueError(f'Protein-Sol produced an empty sequence id: {raw_sequence_id!r}')
    return sequence_id


def _parse_scaled_sol_scores(prediction_path):
    rows = {}
    with open(prediction_path, newline='') as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) >= 4 and row[0] == 'SEQUENCE PREDICTIONS':
                sequence_id = _normalize_sequence_id(row[1])
                if sequence_id in rows:
                    raise ValueError(
                        'Protein-Sol produced duplicate sequence ids in '
                        f'{prediction_path}: {sequence_id!r}'
                    )
                rows[sequence_id] = float(row[3])
    if not rows:
        raise ValueError(f'Protein-Sol produced no SEQUENCE PREDICTIONS rows in {prediction_path}')
    return rows


def score_developability_components(proteinsol_scores, sequences):
    if len(proteinsol_scores) != len(sequences):
        raise ValueError('proteinsol_scores length must match sequences length')
    solubility = []
    liability = []
    developability = []
    for raw_score, sequence in zip(proteinsol_scores, sequences):
        sol = max(0.0, min(1.0, float(raw_score)))
        liab = float(liability_reward(sequence))
        solubility.append(sol)
        liability.append(liab)
        developability.append(0.8 * sol + 0.2 * liab)
    return {
        'solubility': solubility,
        'liability_reward': liability,
        'developability': developability,
    }


class ProteinSolScorer:
    def __init__(self, model_name_or_path, tokenizer_name_or_path=None, device='cpu', batch_size=16):
        if tokenizer_name_or_path is not None:
            raise ValueError('Protein-Sol uses the official CLI bundle; tokenizer_name_or_path must be omitted')
        if device not in {'cpu', 'cuda'}:
            raise ValueError(f'Protein-Sol device must be cpu or cuda, got {device!r}')
        if not model_name_or_path:
            raise ValueError('Protein-Sol model_name_or_path is required')
        self.batch_size = validate_batch_size(batch_size, field_name='developability.batch_size')
        self.device = device
        self.bundle_root = _resolve_proteinsol_root(model_name_or_path)
        self._workspace = None
        self._workspace_root = None
        self.last_move_to_device_sec = 0.0
        self.last_release_to_cpu_sec = 0.0

    def _ensure_workspace(self):
        if self._workspace_root is not None:
            return
        workspace = tempfile.TemporaryDirectory(prefix='proteinsol_')
        destination = Path(workspace.name) / 'protein-sol-sequence-prediction-software'
        shutil.copytree(self.bundle_root, destination)
        self._workspace = workspace
        self._workspace_root = destination

    def release(self):
        return

    def _write_fasta(self, sequence_items):
        fasta_path = self._workspace_root / 'batch.fasta'
        with open(fasta_path, 'w') as handle:
            for sequence_id, sequence in sequence_items:
                if not sequence:
                    raise ValueError('Protein-Sol sequences must be non-empty')
                handle.write(f'>{sequence_id}\n{sequence}\n')
        return fasta_path

    def _clear_previous_outputs(self):
        for name in [
            'batch.fasta',
            'batch.fasta_ORIGINAL',
            'seq_prediction.txt',
            'seq_composition.txt',
            'run.log',
            'blah.txt',
        ]:
            path = self._workspace_root / name
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()

    def _run_bundle(self, sequence_items):
        self._clear_previous_outputs()
        fasta_path = self._write_fasta(sequence_items)
        result = subprocess.run(
            ['bash', 'multiple_prediction_wrapper_export.sh', fasta_path.name],
            cwd=self._workspace_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                'Protein-Sol scoring failed with non-zero exit status '
                f'{result.returncode}: stdout={result.stdout!r} stderr={result.stderr!r}'
            )
        return _parse_scaled_sol_scores(self._workspace_root / 'seq_prediction.txt')

    def _score_missing_individually(self, missing_ids, id_to_sequence):
        recovered_scores = {}
        for sequence_id in missing_ids:
            score_map = self._run_bundle([(sequence_id, id_to_sequence[sequence_id])])
            if set(score_map) != {sequence_id}:
                raise RuntimeError(
                    'Protein-Sol individual retry returned unexpected sequence ids: '
                    f'expected only {sequence_id!r}, got {sorted(score_map)}'
                )
            recovered_scores[sequence_id] = score_map[sequence_id]
        return recovered_scores

    def score_raw(self, sequences):
        if not sequences:
            raise ValueError('sequences must be non-empty')
        self._ensure_workspace()
        outputs = []
        for chunk in iter_chunks(sequences, self.batch_size):
            sequence_items = [(f'seq_{idx}', sequence) for idx, sequence in enumerate(chunk)]
            expected_ids = [sequence_id for sequence_id, _ in sequence_items]
            id_to_sequence = dict(sequence_items)
            chunk_scores = self._run_bundle(sequence_items)
            unexpected_ids = sorted(set(chunk_scores) - set(expected_ids))
            if unexpected_ids:
                raise RuntimeError(
                    'Protein-Sol returned sequence ids that were not present in the input chunk: '
                    f'{unexpected_ids}'
                )
            missing_ids = [sequence_id for sequence_id in expected_ids if sequence_id not in chunk_scores]
            if missing_ids:
                recovered_scores = self._score_missing_individually(missing_ids, id_to_sequence)
                chunk_scores.update(recovered_scores)
                remaining_missing = [sequence_id for sequence_id in expected_ids if sequence_id not in chunk_scores]
                if remaining_missing:
                    raise RuntimeError(
                        'Protein-Sol failed to return scores for some sequences after individual retry: '
                        f'{remaining_missing}'
                    )
            outputs.extend(chunk_scores[sequence_id] for sequence_id in expected_ids)
        return outputs


def developability_reward(proteinsol_scores, sequences):
    return score_developability_components(proteinsol_scores, sequences)['developability']
