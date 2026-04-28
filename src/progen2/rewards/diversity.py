from functools import lru_cache

from progen2.rewards.common import is_valid_protein_sequence


def _validate_sequence(sequence):
    sequence = str(sequence).strip().upper()
    if not sequence:
        raise ValueError('protein sequence must be non-empty')
    return sequence


@lru_cache(maxsize=4096)
def normalized_edit_similarity(sequence_a, sequence_b):
    sequence_a = _validate_sequence(sequence_a)
    sequence_b = _validate_sequence(sequence_b)
    len_a = len(sequence_a)
    len_b = len(sequence_b)
    dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]
    for i in range(len_a + 1):
        dp[i][0] = i
    for j in range(len_b + 1):
        dp[0][j] = j
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            substitution_cost = 0 if sequence_a[i - 1] == sequence_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + substitution_cost,
            )
    distance = dp[len_a][len_b]
    return 1.0 - (distance / float(max(len_a, len_b)))


def compute_group_diversity_reward(sequences):
    if len(sequences) < 2:
        raise ValueError('group diversity requires at least two sequences')
    similarities = []
    for left_idx in range(len(sequences)):
        for right_idx in range(left_idx + 1, len(sequences)):
            similarities.append(normalized_edit_similarity(sequences[left_idx], sequences[right_idx]))
    if not similarities:
        raise ValueError('group diversity requires at least one pairwise comparison')
    return 1.0 - (sum(similarities) / float(len(similarities)))


def compute_group_diversity_reward_or_zero(sequences):
    valid_sequences = [
        str(sequence).strip().upper()
        for sequence in sequences
        if is_valid_protein_sequence(sequence)
    ]
    if len(valid_sequences) < 2:
        return 0.0
    return compute_group_diversity_reward(valid_sequences)


def compute_group_diversity_loo_credits(sequences):
    if len(sequences) < 2:
        raise ValueError('LOO diversity credit requires at least two rollouts')
    full_diversity = compute_group_diversity_reward_or_zero(sequences)
    credits = []
    for remove_idx in range(len(sequences)):
        reduced = sequences[:remove_idx] + sequences[remove_idx + 1:]
        credits.append(full_diversity - compute_group_diversity_reward_or_zero(reduced))
    return credits
