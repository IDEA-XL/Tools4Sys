import unittest

import torch

from genmol.rl.lead_cpgrpo import get_per_token_logps_full


class LeadCpGrpoCoreTest(unittest.TestCase):
    def test_get_per_token_logps_full_shape(self):
        input_ids = torch.tensor(
            [
                [[1, 5, 9, 6, 2]],
                [[1, 5, 9, 6, 2]],
            ],
            dtype=torch.long,
        )
        completion_mask = torch.tensor([[False, False, True, False, False]])

        def score_fn(batch):
            logits = torch.zeros(batch.size(0), batch.size(1), 16)
            logits.scatter_(2, batch.unsqueeze(-1), 5.0)
            return logits

        logps = get_per_token_logps_full(
            score_fn=score_fn,
            input_ids=input_ids,
            completion_mask=completion_mask,
            mask_token_id=9,
            mask_seeds=[3, 4],
            gradient_accumulation_steps=1,
            requires_grad=False,
        )
        self.assertEqual(tuple(logps.shape), (1, 2, 5))


if __name__ == '__main__':
    unittest.main()
