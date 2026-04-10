from dataclasses import dataclass
from functools import lru_cache

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from genmol.rl.reward import MolecularReward, RewardRecord


@dataclass(frozen=True)
class LeadRewardRecord:
    reward: float
    is_valid: bool
    alert_hit: bool
    qed: float | None
    sa: float | None
    sa_score: float | None
    soft_reward: float | None
    sim: float | None
    smiles: str | None
    seed_smiles: str


@lru_cache(maxsize=100000)
def _canonical_seed_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        raise ValueError(f'Invalid seed smiles for lead optimization: {smiles}')
    return Chem.MolToSmiles(mol)


@lru_cache(maxsize=100000)
def _seed_fp(smiles):
    canonical = _canonical_seed_smiles(smiles)
    mol = Chem.MolFromSmiles(canonical, sanitize=True)
    if mol is None:
        raise ValueError(f'Failed to rebuild canonical seed smiles: {smiles}')
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)


def compute_similarity(seed_smiles, candidate_smiles):
    if candidate_smiles is None:
        return None
    candidate_mol = Chem.MolFromSmiles(candidate_smiles, sanitize=True)
    if candidate_mol is None:
        return None
    candidate_fp = AllChem.GetMorganFingerprintAsBitVect(candidate_mol, 2, 2048)
    return float(DataStructs.TanimotoSimilarity(_seed_fp(seed_smiles), candidate_fp))


class LeadOptimizationReward:
    def __init__(self, sim_weight=1.0):
        self.sim_weight = float(sim_weight)
        self.base_reward = MolecularReward()

    @property
    def num_workers(self):
        return self.base_reward.num_workers

    def close(self):
        self.base_reward.close()

    def _combine(self, seed_smiles, record: RewardRecord):
        canonical_seed = _canonical_seed_smiles(seed_smiles)
        if not record.is_valid or record.smiles is None:
            return LeadRewardRecord(
                reward=-1.0,
                is_valid=False,
                alert_hit=False,
                qed=record.qed,
                sa=record.sa,
                sa_score=record.sa_score,
                soft_reward=record.soft_reward,
                sim=None,
                smiles=record.smiles,
                seed_smiles=canonical_seed,
            )

        sim = compute_similarity(canonical_seed, record.smiles)
        if sim is None:
            return LeadRewardRecord(
                reward=-1.0,
                is_valid=False,
                alert_hit=False,
                qed=record.qed,
                sa=record.sa,
                sa_score=record.sa_score,
                soft_reward=record.soft_reward,
                sim=None,
                smiles=record.smiles,
                seed_smiles=canonical_seed,
            )

        # Assumption: "de novo reward + SIM reward" means adding similarity to the
        # fully gated de novo reward for valid molecules.
        total_reward = float(record.reward) + self.sim_weight * float(sim)
        return LeadRewardRecord(
            reward=total_reward,
            is_valid=True,
            alert_hit=record.alert_hit,
            qed=record.qed,
            sa=record.sa,
            sa_score=record.sa_score,
            soft_reward=record.soft_reward,
            sim=sim,
            smiles=record.smiles,
            seed_smiles=canonical_seed,
        )

    def score(self, seed_smiles_list, candidate_smiles_list):
        if len(seed_smiles_list) != len(candidate_smiles_list):
            raise ValueError(
                'seed_smiles_list and candidate_smiles_list must have the same length: '
                f'{len(seed_smiles_list)} vs {len(candidate_smiles_list)}'
            )
        base_records = self.base_reward.score(candidate_smiles_list)
        return [self._combine(seed_smiles, record) for seed_smiles, record in zip(seed_smiles_list, base_records)]
