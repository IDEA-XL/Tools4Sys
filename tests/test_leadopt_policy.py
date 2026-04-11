import random
import unittest

from genmol.rl.lead_policy import select_valid_mutation_interval


class LeadOptPolicyTest(unittest.TestCase):
    def test_select_valid_mutation_interval_avoids_zero_width_when_context_is_full(self):
        rng = random.Random(123)
        mask_start_idx, mask_end_idx, max_insert = select_valid_mutation_interval(
            special_token_idx=[0, 1, 5, 7],
            seed_length=8,
            max_position_embeddings=8,
            rng=rng,
        )
        self.assertGreater(mask_end_idx - mask_start_idx, 0)
        self.assertGreaterEqual(max_insert, 1)

    def test_select_valid_mutation_interval_raises_when_no_room_exists(self):
        rng = random.Random(123)
        with self.assertRaisesRegex(ValueError, 'No valid mutation interval'):
            select_valid_mutation_interval(
                special_token_idx=[0, 1, 2],
                seed_length=3,
                max_position_embeddings=3,
                rng=rng,
            )


if __name__ == '__main__':
    unittest.main()
