from dataclasses import dataclass


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


class MolecularReward:
    def __init__(self):
        from rdkit import Chem
        from rdkit.Chem import QED
        from tdc import Oracle
        import tdc

        self._chem = Chem
        self._qed = QED
        self._sa_oracle = Oracle('sa')
        self._filter = tdc.chem_utils.oracle.filter.MolFilter(
            filters=['PAINS', 'SureChEMBL', 'Glaxo'],
            property_filters_flag=False,
        )

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

    def score(self, smiles_list):
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
            sa_scores = [float(score) for score in self._sa_oracle(canonical_smiles)]

            for index, smiles, qed_score, sa_score in zip(valid_indices, canonical_smiles, qed_scores, sa_scores):
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
