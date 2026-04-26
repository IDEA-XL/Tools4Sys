import unittest

import torch

from genmol.rl.cpgrpo import (
    compute_clipped_grpo_loss,
    compute_grouped_advantages,
    forward_process,
    get_per_token_logps,
    split_tensor_dict,
)
from genmol.rl.reward import (
    apply_reward_gate,
    compute_internal_diversity,
    compute_internal_diversity_loo_credits,
    compute_soft_reward,
    sa_to_score,
)


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

    def test_internal_diversity_loo_credits_match_naive_formula(self):
        smiles = ['CCO', 'CCN', 'c1ccccc1', None, 'not-a-smiles']
        full_diversity = compute_internal_diversity(smiles)
        expected = []
        for remove_idx in range(len(smiles)):
            reduced = smiles[:remove_idx] + smiles[remove_idx + 1:]
            expected.append(full_diversity - compute_internal_diversity(reduced))

        actual = compute_internal_diversity_loo_credits(smiles)
        self.assertEqual(len(actual), len(smiles))
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right)

    def test_split_tensor_dict(self):
        payload = {
            'x': torch.arange(8).view(4, 2),
            'y': torch.arange(4),
            'meta': [1, 2, 3],
        }
        chunks = split_tensor_dict(payload, 2)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(torch.equal(chunks[0]['x'], torch.tensor([[0, 1], [2, 3]])))
        self.assertTrue(torch.equal(chunks[1]['y'], torch.tensor([2, 3])))
        self.assertEqual(chunks[0]['meta'], [1, 2, 3])

    def test_forward_process_masks_completion_only(self):
        batch = torch.tensor([[11, 12, 13, 14, 0]])
        completion_mask = torch.tensor([[False, True, True, False, False]])
        noisy, _, partial_mask = forward_process(
            batch=batch,
            completion_mask=completion_mask,
            mask_id=99,
            seed=7,
            gradient_accumulation_steps=1,
            accumulate=False,
        )
        self.assertTrue(torch.equal(noisy[0][0, [0, 3, 4]], batch[0, [0, 3, 4]]))
        self.assertTrue(torch.equal(partial_mask & (~completion_mask), torch.zeros_like(completion_mask)))

    def test_grouped_advantages_leave_one_out(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        advantages, stds, zero_std_ratio = compute_grouped_advantages(rewards, num_generations=2, scale_rewards=False)
        expected = torch.tensor([-1.0, 1.0, -1.0, 1.0])
        self.assertTrue(torch.allclose(advantages, expected))
        self.assertEqual(stds.shape[0], 4)
        self.assertEqual(zero_std_ratio, 0.0)

    def test_get_per_token_logps_shape(self):
        input_ids = torch.tensor(
            [
                [[1, 5, 6, 2]],
                [[1, 5, 6, 2]],
            ],
            dtype=torch.long,
        )
        completion_mask = torch.tensor([[False, True, True, False]])

        def score_fn(batch):
            logits = torch.zeros(batch.size(0), batch.size(1), 10)
            logits.scatter_(2, batch.unsqueeze(-1), 5.0)
            return logits

        logps = get_per_token_logps(
            score_fn=score_fn,
            input_ids=input_ids,
            completion_mask=completion_mask,
            mask_token_id=9,
            mask_seeds=[3, 4],
            gradient_accumulation_steps=1,
            requires_grad=False,
        )
        self.assertEqual(tuple(logps.shape), (1, 2, 4))

    def test_clipped_loss_reports_unit_ratio_before_update(self):
        old_log_probs = torch.tensor([[[0.0, -0.1]]])
        new_log_probs = old_log_probs.clone()
        advantages = torch.tensor([1.0])
        completion_mask = torch.tensor([[True, True]])
        _, metrics = compute_clipped_grpo_loss(
            new_log_probs=new_log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            completion_mask=completion_mask,
            epsilon=0.5,
        )
        self.assertAlmostEqual(metrics['ratio_mean'], 1.0)
        self.assertAlmostEqual(metrics['clip_ratio_region_mean'], 0.0)


if __name__ == '__main__':
    unittest.main()
