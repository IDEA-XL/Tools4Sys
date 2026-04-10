import unittest

import torch
import torch.nn.functional as F

from genmol.rl.lead_cpgrpo import (
    _selective_log_softmax_materialized,
    get_per_token_logps_full,
)


def _reference_selective_log_softmax(logits, index, weights, mask):
    full_batch_size = logits.size(0) // 3
    num_iterations = weights.size(0) // 3
    batch_size = full_batch_size // num_iterations
    per_token_logps = []

    for sample_idx in range(full_batch_size):
        labels = index[sample_idx].clone()
        chunk_idx, offset = divmod(sample_idx, batch_size)
        base = chunk_idx * 3 * batch_size
        logits_index = torch.tensor(
            [base + offset, base + batch_size + offset, base + 2 * batch_size + offset],
            device=logits.device,
        )
        seq_logits = logits[logits_index]
        seq_logps = F.log_softmax(seq_logits, dim=-1)
        gathered = seq_logps.gather(
            dim=-1,
            index=labels.unsqueeze(0).unsqueeze(-1).expand(3, -1, 1),
        ).squeeze(-1)
        seq_weights = weights[chunk_idx * 3:(chunk_idx + 1) * 3]
        seq_mask = mask[sample_idx].clone()
        weighted = torch.where(seq_mask, gathered[1] * seq_weights[1], gathered[2] * seq_weights[2])
        per_token_logps.append((gathered[0] + weighted) / 2)

    return torch.stack(per_token_logps, dim=0)


class LeadCpGrpoCoreTest(unittest.TestCase):
    def test_materialized_selective_log_softmax_matches_reference(self):
        torch.manual_seed(7)
        full_batch_size = 8
        seq_len = 61
        vocab_size = 19
        logits = torch.randn(full_batch_size * 3, seq_len, vocab_size)
        index = torch.randint(0, vocab_size, (full_batch_size, seq_len), dtype=torch.long)
        mask = torch.randint(0, 2, (full_batch_size, seq_len), dtype=torch.bool)
        weights = torch.tensor([1.0, 1.25, 0.8, 1.0, 1.4, 0.6], dtype=torch.float32)

        expected = _reference_selective_log_softmax(
            logits=logits,
            index=index,
            weights=weights,
            mask=mask,
        )
        actual = _selective_log_softmax_materialized(
            logits=logits,
            index=index,
            weights=weights,
            mask=mask,
        )

        self.assertTrue(torch.allclose(actual, expected))

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

    def test_chunked_and_unchunked_outputs_match(self):
        input_ids = torch.tensor(
            [
                [[1, 5, 9, 6, 2], [1, 7, 9, 8, 2]],
                [[1, 5, 9, 6, 2], [1, 7, 9, 8, 2]],
            ],
            dtype=torch.long,
        )
        completion_mask = torch.tensor(
            [
                [False, False, True, False, False],
                [False, False, True, False, False],
            ]
        )

        def score_fn(batch):
            logits = torch.zeros(batch.size(0), batch.size(1), 16)
            logits.scatter_(2, batch.unsqueeze(-1), 5.0)
            return logits

        full = get_per_token_logps_full(
            score_fn=score_fn,
            input_ids=input_ids,
            completion_mask=completion_mask,
            mask_token_id=9,
            mask_seeds=[3, 4],
            gradient_accumulation_steps=1,
            requires_grad=False,
        )
        chunked = get_per_token_logps_full(
            score_fn=score_fn,
            input_ids=input_ids,
            completion_mask=completion_mask,
            mask_token_id=9,
            mask_seeds=[3, 4],
            gradient_accumulation_steps=1,
            requires_grad=False,
            score_chunk_size=1,
        )
        self.assertTrue(torch.allclose(full, chunked))


if __name__ == '__main__':
    unittest.main()
