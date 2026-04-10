import os
import unittest

from genmol.rl.reward import (
    MolecularReward,
    RewardRecord,
    _AlertFilter,
    apply_reward_gate,
    compute_soft_reward,
    sa_to_score,
)


TEST_SMILES = [
    'CCS(=O)(=O)N1CC(CC#N)(n2cc(-c3ncnc4[nH]ccc34)cn2)C1',
    'NS(=O)(=O)c1cc2c(cc1Cl)NC(C1CC3C=CC1C3)NS2(=O)=O',
    'CCO',
    'CCN',
    'c1ccccc1',
    'CC(=O)O',
    'CCOC(=O)C',
    'CC(C)O',
    'CC(C)N',
    'CCCO',
    'O=C(O)c1ccccc1O',
    'CC1=CC(=O)NC(=O)N1',
    'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
    'C1CCCCC1',
    'N#CC1=CC=CC=C1',
    'C1=CC=C(C=C1)N(=O)=O',
    'CC(C)(C)OC(=O)N1CCC(CC1)NC(=O)C2=CC=CC=C2',
    'CCOC1=CC=CC=C1',
    'not_a_smiles',
    'C1CC',
    None,
    '',
    'CCO',
]


class LegacyMolecularReward:
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


class RewardEquivalenceTest(unittest.TestCase):
    def test_alert_filter_matches_legacy_molfilter(self):
        import tdc

        legacy_filter = tdc.chem_utils.oracle.filter.MolFilter(
            filters=['PAINS', 'SureChEMBL', 'Glaxo'],
            property_filters_flag=False,
        )
        current_filter = _AlertFilter()
        valid_smiles = [smiles for smiles in TEST_SMILES if isinstance(smiles, str) and smiles]
        self.assertEqual(
            list(current_filter(valid_smiles)),
            list(legacy_filter(valid_smiles)),
        )

    def test_single_worker_matches_legacy_reward(self):
        legacy_reward = LegacyMolecularReward()
        os.environ['GENMOL_REWARD_WORKERS'] = '1'
        current_reward = MolecularReward()
        try:
            self.assertEqual(current_reward.score(TEST_SMILES), legacy_reward.score(TEST_SMILES))
        finally:
            current_reward.close()
            os.environ.pop('GENMOL_REWARD_WORKERS', None)

    def test_multi_worker_matches_legacy_reward(self):
        legacy_reward = LegacyMolecularReward()
        os.environ['GENMOL_REWARD_WORKERS'] = '2'
        current_reward = MolecularReward()
        try:
            self.assertEqual(current_reward.score(TEST_SMILES), legacy_reward.score(TEST_SMILES))
        finally:
            current_reward.close()
            os.environ.pop('GENMOL_REWARD_WORKERS', None)


if __name__ == '__main__':
    unittest.main()
