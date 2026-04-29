from __future__ import annotations

import csv
import math
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import SimpleQueue

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


def _normalize_optional_num_workers(num_workers):
    if num_workers is None:
        return None
    return validate_batch_size(num_workers, field_name='developability.num_workers')


def _resolve_env_positive_int(name):
    raw_value = os.environ.get(name)
    if raw_value is None or str(raw_value).strip() == '':
        return None
    try:
        value = int(str(raw_value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a positive integer when set, got {raw_value!r}') from exc
    if value <= 0:
        raise ValueError(f'{name} must be a positive integer when set, got {raw_value!r}')
    return value


def _resolve_available_cpu_budget():
    slurm_cpus_per_task = _resolve_env_positive_int('SLURM_CPUS_PER_TASK')
    if slurm_cpus_per_task is not None:
        return slurm_cpus_per_task
    if hasattr(os, 'sched_getaffinity'):
        affinity = os.sched_getaffinity(0)
        if affinity:
            return len(affinity)
    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count <= 0:
        raise RuntimeError('Unable to resolve available CPU budget for Protein-Sol worker auto-scaling')
    return int(cpu_count)


def _resolve_local_world_size():
    local_world_size = _resolve_env_positive_int('LOCAL_WORLD_SIZE')
    if local_world_size is not None:
        return local_world_size
    return 1


def _default_num_workers():
    cpu_budget = _resolve_available_cpu_budget()
    local_world_size = _resolve_local_world_size()
    return max(1, cpu_budget // local_world_size)


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
    def __init__(self, model_name_or_path, tokenizer_name_or_path=None, device='cpu', batch_size=16, num_workers=None):
        if tokenizer_name_or_path is not None:
            raise ValueError('Protein-Sol uses the official CLI bundle; tokenizer_name_or_path must be omitted')
        if device not in {'cpu', 'cuda'}:
            raise ValueError(f'Protein-Sol device must be cpu or cuda, got {device!r}')
        if not model_name_or_path:
            raise ValueError('Protein-Sol model_name_or_path is required')
        self.batch_size = validate_batch_size(batch_size, field_name='developability.batch_size')
        self.num_workers = _normalize_optional_num_workers(num_workers)
        self.device = device
        self.bundle_root = _resolve_proteinsol_root(model_name_or_path)
        self._workers = []
        self.last_move_to_device_sec = 0.0
        self.last_release_to_cpu_sec = 0.0
        self.last_num_workers = 0
        self.last_effective_batch_size = 0
        self.last_chunk_count = 0

    def _resolved_num_workers(self):
        if self.num_workers is not None:
            return self.num_workers
        return _default_num_workers()

    def _resolve_runtime_worker_count(self, num_sequences):
        return max(1, min(self._resolved_num_workers(), int(num_sequences)))

    def _resolve_effective_batch_size(self, num_sequences, worker_count):
        if num_sequences <= 0:
            raise ValueError(f'num_sequences must be positive, got {num_sequences}')
        if worker_count <= 0:
            raise ValueError(f'worker_count must be positive, got {worker_count}')
        # Cap the chunk size so one rank can fan out across its available workers in a single wave.
        per_worker_target = max(1, math.ceil(num_sequences / worker_count))
        return min(self.batch_size, per_worker_target)

    def _ensure_workers(self, worker_count):
        while len(self._workers) < worker_count:
            self._workers.append(_ProteinSolWorker(self.bundle_root))

    def release(self):
        return

    def _score_chunk_with_worker(self, worker_index, chunk):
        return self._workers[worker_index].score_chunk(chunk)

    def score_raw(self, sequences):
        if not sequences:
            raise ValueError('sequences must be non-empty')
        worker_count = self._resolve_runtime_worker_count(len(sequences))
        effective_batch_size = self._resolve_effective_batch_size(len(sequences), worker_count)
        chunks = list(iter_chunks(sequences, effective_batch_size))
        self._ensure_workers(worker_count)
        self.last_num_workers = worker_count
        self.last_effective_batch_size = effective_batch_size
        self.last_chunk_count = len(chunks)
        if len(chunks) == 1:
            return self._score_chunk_with_worker(0, chunks[0])

        outputs = [None] * len(chunks)
        max_workers = min(worker_count, len(chunks))
        idle_workers = SimpleQueue()
        for worker_index in range(max_workers):
            idle_workers.put(worker_index)

        def _run_chunk(chunk):
            worker_index = idle_workers.get()
            try:
                return self._score_chunk_with_worker(worker_index, chunk)
            finally:
                idle_workers.put(worker_index)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {}
            for chunk_index, chunk in enumerate(chunks):
                future = executor.submit(_run_chunk, chunk)
                future_to_chunk[future] = chunk_index
            for future in as_completed(future_to_chunk):
                chunk_index = future_to_chunk[future]
                outputs[chunk_index] = future.result()

        flattened = []
        for chunk_scores in outputs:
            flattened.extend(chunk_scores)
        return flattened


class _ProteinSolWorker:
    def __init__(self, bundle_root):
        self.bundle_root = Path(bundle_root).resolve()
        workspace = tempfile.TemporaryDirectory(prefix='proteinsol_')
        destination = Path(workspace.name) / 'protein-sol-sequence-prediction-software'
        shutil.copytree(self.bundle_root, destination)
        self._workspace = workspace
        self._workspace_root = destination
        self._subprocess_env = dict(os.environ)
        self._subprocess_env['OMP_NUM_THREADS'] = '1'
        self._subprocess_env['MKL_NUM_THREADS'] = '1'
        self._subprocess_env['OPENBLAS_NUM_THREADS'] = '1'
        self._subprocess_env['NUMEXPR_NUM_THREADS'] = '1'

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
            env=self._subprocess_env,
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

    def score_chunk(self, chunk):
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
        return [chunk_scores[sequence_id] for sequence_id in expected_ids]


def developability_reward(proteinsol_scores, sequences):
    return score_developability_components(proteinsol_scores, sequences)['developability']
