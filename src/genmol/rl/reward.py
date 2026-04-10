import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass


_PROCESS_KERNEL = None


def sa_to_score(sa_value):
    return max(0.0, min((6.0 - float(sa_value)) / 5.0, 1.0))


def compute_soft_reward(qed_value, sa_value):
    return 0.6 * float(qed_value) + 0.4 * sa_to_score(sa_value)


def apply_reward_gate(soft_reward, is_valid, alert_hit):
    if not is_valid:
        return -1.0
    if alert_hit:
        return 0.2 * float(soft_reward)
    return float(soft_reward)


@dataclass(frozen=True)
class RewardRecord:
    reward: float
    is_valid: bool
    alert_hit: bool
    qed: float | None
    sa: float | None
    sa_score: float | None
    soft_reward: float | None
    smiles: str | None


class _AlertFilter:
    def __init__(self):
        from rd_filters.rd_filters import RDFilters, read_rules
        import pandas as pd
        import pkg_resources

        alert_file_name = pkg_resources.resource_filename(
            'rd_filters',
            'data/alert_collection.csv',
        )
        rules_file_path = pkg_resources.resource_filename(
            'rd_filters',
            'data/rules.json',
        )
        self._rf = RDFilters(alert_file_name)
        self._pd = pd
        rule_dict = read_rules(rules_file_path)
        rule_dict['Rule_Inpharmatica'] = False
        rule_dict['Rule_PAINS'] = True
        rule_dict['Rule_SureChEMBL'] = True
        rule_dict['Rule_Glaxo'] = True
        for key in ['HBA', 'HBD', 'LogP', 'MW', 'Rot', 'TPSA']:
            rule_dict.pop(key, None)
        rule_list = [
            key.replace('Rule_', '')
            for key, enabled in rule_dict.items()
            if key.startswith('Rule_') and enabled
        ]
        self._rf.build_rule_list(rule_list)

    def __call__(self, input_data):
        if isinstance(input_data, str):
            input_data = [input_data]
        elif not isinstance(input_data, list):
            raise ValueError('Input must be a list of SMILES or one SMILES string')

        indexed_smiles = list(zip(input_data, list(range(len(input_data)))))
        results = [self._rf.evaluate(item) for item in indexed_smiles]
        frame = self._pd.DataFrame(
            results,
            columns=[
                'SMILES',
                'NAME',
                'FILTER',
                'MW',
                'LogP',
                'HBD',
                'HBA',
                'TPSA',
                'Rot',
            ],
        )
        return frame[frame.FILTER == 'OK'].SMILES.values


def _resolve_reward_workers():
    raw_workers = os.environ.get('GENMOL_REWARD_WORKERS')
    if raw_workers is not None:
        try:
            workers = int(raw_workers)
        except ValueError as exc:
            raise ValueError(f'GENMOL_REWARD_WORKERS must be an integer, got: {raw_workers}') from exc
        if workers <= 0:
            raise ValueError(f'GENMOL_REWARD_WORKERS must be positive, got: {workers}')
        return workers

    raw_total_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if raw_total_cpus is None:
        total_cpus = os.cpu_count() or 1
    else:
        try:
            total_cpus = int(raw_total_cpus)
        except ValueError as exc:
            raise ValueError(f'SLURM_CPUS_PER_TASK must be an integer, got: {raw_total_cpus}') from exc
    if total_cpus <= 0:
        raise ValueError(f'Invalid CPU count for reward workers: {total_cpus}')

    raw_local_world_size = os.environ.get('LOCAL_WORLD_SIZE') or os.environ.get('WORLD_SIZE') or '1'
    try:
        local_world_size = int(raw_local_world_size)
    except ValueError as exc:
        raise ValueError(f'LOCAL_WORLD_SIZE/WORLD_SIZE must be an integer, got: {raw_local_world_size}') from exc
    if local_world_size <= 0:
        raise ValueError(f'Invalid local world size for reward workers: {local_world_size}')

    # Leave one CPU core for the trainer process when possible.
    return max(1, (total_cpus // local_world_size) - 1)


class _RewardKernel:
    def __init__(self):
        from rdkit import Chem
        from rdkit.Chem import QED
        from tdc import Oracle

        self._chem = Chem
        self._qed = QED
        self._sa_oracle = Oracle('sa')
        self._filter = _AlertFilter()

    def _safe_sa_score(self, smiles):
        try:
            return float(self._sa_oracle([smiles])[0])
        except Exception:
            return None

    def _canonicalize(self, smiles):
        if smiles is None:
            return None, None
        try:
            mol = self._chem.MolFromSmiles(smiles, sanitize=True)
        except Exception:
            return None, None
        if mol is None:
            return None, None
        try:
            canonical = self._chem.MolToSmiles(mol)
        except Exception:
            return None, None
        return canonical, mol

    def score_chunk(self, smiles_list):
        canonical_smiles = []
        mols = []
        valid_indices = []
        records = [None] * len(smiles_list)

        for idx, smiles in enumerate(smiles_list):
            canonical, mol = self._canonicalize(smiles)
            if canonical is None or mol is None:
                records[idx] = RewardRecord(
                    reward=-1.0,
                    is_valid=False,
                    alert_hit=False,
                    qed=None,
                    sa=None,
                    sa_score=None,
                    soft_reward=None,
                    smiles=None,
                )
                continue
            canonical_smiles.append(canonical)
            mols.append(mol)
            valid_indices.append(idx)

        if valid_indices:
            pass_smiles = set(self._filter(canonical_smiles))
            qed_scores = [float(self._qed.qed(mol)) for mol in mols]
            sa_scores = [self._safe_sa_score(smiles) for smiles in canonical_smiles]

            for index, smiles, qed_score, sa_score in zip(valid_indices, canonical_smiles, qed_scores, sa_scores):
                if sa_score is None:
                    records[index] = RewardRecord(
                        reward=-1.0,
                        is_valid=False,
                        alert_hit=False,
                        qed=qed_score,
                        sa=None,
                        sa_score=None,
                        soft_reward=None,
                        smiles=smiles,
                    )
                    continue
                alert_hit = smiles not in pass_smiles
                soft_reward = compute_soft_reward(qed_score, sa_score)
                records[index] = RewardRecord(
                    reward=apply_reward_gate(soft_reward, is_valid=True, alert_hit=alert_hit),
                    is_valid=True,
                    alert_hit=alert_hit,
                    qed=qed_score,
                    sa=sa_score,
                    sa_score=sa_to_score(sa_score),
                    soft_reward=soft_reward,
                    smiles=smiles,
                )

        return records


def _initialize_process_reward_kernel():
    global _PROCESS_KERNEL
    _PROCESS_KERNEL = _RewardKernel()


def _score_reward_chunk(smiles_chunk):
    if _PROCESS_KERNEL is None:
        raise RuntimeError('Reward worker kernel is not initialized')
    return _PROCESS_KERNEL.score_chunk(smiles_chunk)


class MolecularReward:
    def __init__(self):
        self.num_workers = _resolve_reward_workers()
        self._kernel = _RewardKernel()
        self._pool = None
        if self.num_workers > 1:
            self._pool = ProcessPoolExecutor(
                max_workers=self.num_workers,
                mp_context=mp.get_context('spawn'),
                initializer=_initialize_process_reward_kernel,
            )

    def close(self):
        if self._pool is None:
            return
        self._pool.shutdown(wait=True, cancel_futures=False)
        self._pool = None

    def score(self, smiles_list):
        if not smiles_list:
            return []
        if self._pool is None or len(smiles_list) == 1:
            return self._kernel.score_chunk(smiles_list)

        chunk_count = min(self.num_workers, len(smiles_list))
        chunk_size = math.ceil(len(smiles_list) / chunk_count)
        futures = []
        for start in range(0, len(smiles_list), chunk_size):
            futures.append(self._pool.submit(_score_reward_chunk, smiles_list[start:start + chunk_size]))

        records = []
        for future in futures:
            records.extend(future.result())

        if len(records) != len(smiles_list):
            raise RuntimeError(
                f'Reward worker returned mismatched record count: expected {len(smiles_list)}, got {len(records)}'
            )
        return records
