import os
import tempfile
import unittest

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import safe as sf

from genmol.rl.lead_reward import LeadOptimizationReward, compute_similarity
from genmol.rl.lead_specs import load_seed_smiles, sample_seed_smiles
from genmol.rl.reward import MolecularReward


SEED_SMILES = 'CCO'
CANDIDATES = ['CCO', 'CCN', 'c1ccccc1', 'not_a_smiles', None]


class LeadOptimizationRewardTest(unittest.TestCase):
    def test_similarity_matches_official_tanimoto_logic(self):
        seed_mol = Chem.MolFromSmiles(SEED_SMILES)
        seed_fp = AllChem.GetMorganFingerprintAsBitVect(seed_mol, 2, 2048)
        candidate_mol = Chem.MolFromSmiles('CCN')
        candidate_fp = AllChem.GetMorganFingerprintAsBitVect(candidate_mol, 2, 2048)
        expected = float(DataStructs.BulkTanimotoSimilarity(seed_fp, [candidate_fp])[0])
        self.assertAlmostEqual(compute_similarity(SEED_SMILES, 'CCN'), expected)

    def test_lead_reward_adds_similarity_on_top_of_denovo_reward(self):
        os.environ['GENMOL_REWARD_WORKERS'] = '1'
        try:
            base_reward = MolecularReward()
            lead_reward = LeadOptimizationReward(sim_weight=1.0)
        except ImportError as exc:
            os.environ.pop('GENMOL_REWARD_WORKERS', None)
            self.skipTest(f'reward dependencies unavailable in local test environment: {exc}')
        try:
            base_records = base_reward.score(CANDIDATES)
            lead_records = lead_reward.score([SEED_SMILES] * len(CANDIDATES), CANDIDATES)

            self.assertEqual(len(base_records), len(lead_records))
            for base_record, lead_record, candidate in zip(base_records, lead_records, CANDIDATES):
                self.assertEqual(base_record.is_valid, lead_record.is_valid)
                self.assertEqual(base_record.alert_hit, lead_record.alert_hit)
                self.assertEqual(base_record.qed, lead_record.qed)
                self.assertEqual(base_record.sa, lead_record.sa)
                self.assertEqual(base_record.sa_score, lead_record.sa_score)
                self.assertEqual(base_record.soft_reward, lead_record.soft_reward)
                self.assertEqual(lead_record.seed_smiles, Chem.MolToSmiles(Chem.MolFromSmiles(SEED_SMILES)))
                if not base_record.is_valid or candidate is None:
                    self.assertEqual(lead_record.reward, -1.0)
                    self.assertIsNone(lead_record.sim)
                    continue
                expected_sim = compute_similarity(SEED_SMILES, base_record.smiles)
                self.assertAlmostEqual(lead_record.sim, expected_sim)
                self.assertAlmostEqual(lead_record.reward, base_record.reward + expected_sim)
        finally:
            base_reward.close()
            lead_reward.close()
            os.environ.pop('GENMOL_REWARD_WORKERS', None)

    def test_seed_loader_reads_three_parquet_shards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_a = pd.DataFrame({'smiles': ['CCO', 'CCN']})
            frame_b = pd.DataFrame({'smiles': ['c1ccccc1']})
            frame_c = pd.DataFrame({'smiles': ['CCO']})
            frame_a.to_parquet(os.path.join(tmpdir, 'train-00000-of-00102-a.parquet'))
            frame_b.to_parquet(os.path.join(tmpdir, 'train-00001-of-00102-b.parquet'))
            frame_c.to_parquet(os.path.join(tmpdir, 'train-00002-of-00102-c.parquet'))

            smiles = load_seed_smiles(os.path.join(tmpdir, 'train-0000[0-2]-of-00102-*.parquet'))
            self.assertEqual(tuple(smiles), ('CCO', 'CCN', 'c1ccccc1', 'CCO'))

    def test_seed_loader_reads_safe_gpt_input_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            safe_a = 'CCC(=O)CC2.C12=C(C)CCCC1(C)C'
            safe_b = 'CC3.O32.C2C(O)=NC4C.c14cccc(N=C(C)O)c1'
            safe_c = 'CCC3.N3C1CC(C2CCCO2)Oc2cc(Br)ccc21'
            frame_a = pd.DataFrame({'input': [safe_a, safe_b]})
            frame_b = pd.DataFrame({'input': [safe_c]})
            frame_c = pd.DataFrame({'input': [safe_a]})
            frame_a.to_parquet(os.path.join(tmpdir, 'train-00000-of-00102-a.parquet'))
            frame_b.to_parquet(os.path.join(tmpdir, 'train-00001-of-00102-b.parquet'))
            frame_c.to_parquet(os.path.join(tmpdir, 'train-00002-of-00102-c.parquet'))

            smiles = load_seed_smiles(os.path.join(tmpdir, 'train-0000[0-2]-of-00102-*.parquet'))
            self.assertEqual(tuple(smiles), tuple(sf.decode(item) for item in (safe_a, safe_b, safe_c, safe_a)))

    def test_seed_sampler_reads_from_three_shards_without_full_materialization_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_a = pd.DataFrame({'smiles': ['CCO', 'CCN', 'CCC']})
            frame_b = pd.DataFrame({'smiles': ['c1ccccc1', 'CCCl', 'CCBr']})
            frame_c = pd.DataFrame({'smiles': ['CO', 'CN', 'CF']})
            frame_a.to_parquet(os.path.join(tmpdir, 'train-00000-of-00102-a.parquet'), row_group_size=2)
            frame_b.to_parquet(os.path.join(tmpdir, 'train-00001-of-00102-b.parquet'), row_group_size=2)
            frame_c.to_parquet(os.path.join(tmpdir, 'train-00002-of-00102-c.parquet'), row_group_size=2)

            sampled = sample_seed_smiles(
                num_samples=6,
                seed_data_glob=os.path.join(tmpdir, 'train-0000[0-2]-of-00102-*.parquet'),
                seed=123,
            )
            allowed = {'CCO', 'CCN', 'CCC', 'c1ccccc1', 'CCCl', 'CCBr', 'CO', 'CN', 'CF'}
            self.assertEqual(len(sampled), 6)
            self.assertTrue(all(item in allowed for item in sampled))

    def test_seed_loader_skips_empty_decoded_safe_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            valid_safe = 'CCC(=O)CC2.C12=C(C)CCCC1(C)C'
            empty_safe = 'CN=C(N4N3.C4CCC5=O.N15Cc2ccccc2C1.C3C6.O67.C17CCCCCC1.I'
            frame = pd.DataFrame({'input': [empty_safe, valid_safe]})
            frame.to_parquet(os.path.join(tmpdir, 'train-00000-of-00102-a.parquet'))

            smiles = load_seed_smiles(os.path.join(tmpdir, 'train-00000-of-00102-*.parquet'))
            self.assertEqual(tuple(smiles), (sf.decode(valid_safe),))

    def test_seed_sampler_skips_empty_decoded_safe_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            valid_safe = 'CCC(=O)CC2.C12=C(C)CCCC1(C)C'
            empty_safe = 'CN=C(N4N3.C4CCC5=O.N15Cc2ccccc2C1.C3C6.O67.C17CCCCCC1.I'
            frame = pd.DataFrame({'input': [empty_safe, valid_safe]})
            frame.to_parquet(os.path.join(tmpdir, 'train-00000-of-00102-a.parquet'), row_group_size=1)

            smiles = sample_seed_smiles(
                num_samples=4,
                seed_data_glob=os.path.join(tmpdir, 'train-00000-of-00102-*.parquet'),
                seed=123,
            )
            self.assertEqual(tuple(smiles), (sf.decode(valid_safe),) * 4)


if __name__ == '__main__':
    unittest.main()
