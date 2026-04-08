import unittest

import torch

from genmol.rl.cpgrpo import compute_clipped_grpo_loss, compute_leave_one_out_advantages, sample_coupled_masks
from genmol.rl.reward import apply_reward_gate, compute_soft_reward, sa_to_score


class CpGrpoCoreTest(unittest.TestCase):
    def test_sa_to_score_is_clipped(self):
        self.assertEqual(sa_to_score(10.0), 0.0)
        self.assertEqual(sa_to_score(-1.0), 1.0)
        self.assertAlmostEqual(sa_to_score(3.5), 0.5)

    def test_soft_reward_formula(self):
        reward = compute_soft_reward(0.8, 3.5)
        self.assertAlmostEqual(reward, 0.6 * 0.8 + 0.4 * 0.5)

    def test_reward_gate(self):
        self.assertEqual(apply_reward_gate(0.7, is_valid=False, alert_hit=False), -1.0)
        self.assertAlmostEqual(apply_reward_gate(0.7, is_valid=True, alert_hit=True), 0.14)
        self.assertAlmostEqual(apply_reward_gate(0.7, is_valid=True, alert_hit=False), 0.7)

    def test_coupled_masks_cover_completion_once(self):
        completion_mask = torch.tensor([[False, True, True, True, False]])
        _, mask_a, mask_b, _ = sample_coupled_masks(completion_mask, seed=7)
        self.assertTrue(torch.equal(mask_a | mask_b, completion_mask))
        self.assertTrue(torch.equal(mask_a & mask_b, torch.zeros_like(completion_mask)))

    def test_leave_one_out_advantages(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        advantages = compute_leave_one_out_advantages(rewards, group_size=2, scale_rewards=False)
        expected = torch.tensor([-1.0, 1.0, -1.0, 1.0])
        self.assertTrue(torch.allclose(advantages, expected))

    def test_clipped_loss_reports_unit_ratio_before_update(self):
        old_log_probs = torch.tensor([[[0.0, -0.1]], [[0.0, -0.1]]])
        new_log_probs = old_log_probs.clone()
        advantages = torch.tensor([1.0])
        completion_mask = torch.tensor([[True, True]])
        _, metrics = compute_clipped_grpo_loss(
            new_log_probs=new_log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            completion_mask=completion_mask,
            clip_range=0.5,
        )
        self.assertAlmostEqual(metrics['ratio_mean'], 1.0)
        self.assertAlmostEqual(metrics['clip_ratio'], 0.0)


if __name__ == '__main__':
    unittest.main()
