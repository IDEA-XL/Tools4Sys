import unittest

import torch

from genmol.rl.pipeline_reward import combine_seed_rewards, lead_base_reward_from_total, topk_mean


class PipelineRewardTest(unittest.TestCase):
    def test_topk_mean_matches_expected_top2_average(self):
        values = torch.tensor([0.1, 0.5, -1.0, 0.7], dtype=torch.float32)
        self.assertAlmostEqual(topk_mean(values, k=2).item(), 0.6, places=6)

    def test_combine_seed_rewards_matches_weighted_formula(self):
        seed_base = torch.tensor([0.2, 0.8], dtype=torch.float32)
        downstream = torch.tensor([1.0, -1.0], dtype=torch.float32)
        combined = combine_seed_rewards(seed_base, downstream, alpha=0.7)
        expected = torch.tensor([0.76, -0.46], dtype=torch.float32)
        self.assertTrue(torch.allclose(combined, expected))

    def test_lead_base_reward_removes_similarity_only_for_valid_records(self):
        self.assertAlmostEqual(
            lead_base_reward_from_total(total_reward=1.35, similarity=0.4, sim_weight=1.0, is_valid=True),
            0.95,
            places=6,
        )
        self.assertEqual(
            lead_base_reward_from_total(total_reward=-1.0, similarity=None, sim_weight=1.0, is_valid=False),
            -1.0,
        )


if __name__ == '__main__':
    unittest.main()
