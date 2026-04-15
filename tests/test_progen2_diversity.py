from progen2.rewards.diversity import compute_group_diversity_reward, normalized_edit_similarity


def test_normalized_edit_similarity_identical_sequences_is_one():
    assert normalized_edit_similarity('ACDE', 'ACDE') == 1.0


def test_normalized_edit_similarity_detects_difference():
    assert normalized_edit_similarity('AAAA', 'CCCC') == 0.0


def test_group_diversity_reward_is_zero_for_identical_group():
    assert compute_group_diversity_reward(['ACDE', 'ACDE']) == 0.0


def test_group_diversity_reward_positive_for_mixed_group():
    assert compute_group_diversity_reward(['ACDE', 'WCAE']) > 0.0
