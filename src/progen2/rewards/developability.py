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


def _parse_scaled_sol_scores(prediction_path):
    rows = []
    with open(prediction_path, newline='') as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) >= 4 and row[0] == 'SEQUENCE PREDICTIONS':
                rows.append(float(row[3]))
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
        self.bundle_root = _resolve_proteinsol_root(model_name_or_path)
        self._workspace = None
        self._workspace_root = None

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

    def _write_fasta(self, sequences):
        fasta_path = self._workspace_root / 'batch.fasta'
        with open(fasta_path, 'w') as handle:
            for idx, sequence in enumerate(sequences):
                if not sequence:
                    raise ValueError('Protein-Sol sequences must be non-empty')
                handle.write(f'>seq_{idx}\n{sequence}\n')
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

    def score_raw(self, sequences):
        if not sequences:
            raise ValueError('sequences must be non-empty')
        self._ensure_workspace()
        outputs = []
        for chunk in iter_chunks(sequences, self.batch_size):
            self._clear_previous_outputs()
            fasta_path = self._write_fasta(chunk)
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
            chunk_scores = _parse_scaled_sol_scores(self._workspace_root / 'seq_prediction.txt')
            if len(chunk_scores) != len(chunk):
                raise RuntimeError(
                    'Protein-Sol returned a different number of scores than inputs for the current chunk: '
                    f'{len(chunk_scores)} != {len(chunk)}'
                )
            outputs.extend(chunk_scores)
        return outputs


def developability_reward(proteinsol_scores, sequences):
    return score_developability_components(proteinsol_scores, sequences)['developability']
