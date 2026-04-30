import math
from collections import Counter

from progen2.rewards.common import is_valid_protein_sequence, normalize_protein_sequence
from progen2.rewards.diversity import compute_group_diversity_reward, normalized_edit_similarity


def nanmean(values):
    filtered = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not filtered:
        return float('nan')
    return float(sum(filtered) / len(filtered))


def classify_protein_sequence(sequence):
    normalized = normalize_protein_sequence(sequence)
    if not normalized:
        return {'sequence': None, 'is_valid': False, 'invalid_reason': 'empty'}
    if not is_valid_protein_sequence(normalized):
        return {'sequence': normalized, 'is_valid': False, 'invalid_reason': 'unsupported_residue'}
    return {'sequence': normalized, 'is_valid': True, 'invalid_reason': None}


def global_edit_diversity(sequences):
    valid_sequences = [normalize_protein_sequence(sequence) for sequence in sequences if is_valid_protein_sequence(sequence)]
    if len(valid_sequences) < 2:
        return 0.0
    sequence_counts = Counter(valid_sequences)
    unique_sequences = list(sequence_counts.keys())
    total_pairs = len(valid_sequences) * (len(valid_sequences) - 1) // 2
    if total_pairs <= 0:
        return 0.0
    weighted_similarity_sum = 0.0
    for left_idx, left_sequence in enumerate(unique_sequences):
        left_count = sequence_counts[left_sequence]
        if left_count >= 2:
            weighted_similarity_sum += math.comb(left_count, 2) * 1.0
        for right_idx in range(left_idx + 1, len(unique_sequences)):
            right_sequence = unique_sequences[right_idx]
            right_count = sequence_counts[right_sequence]
            weighted_similarity_sum += (
                left_count
                * right_count
                * normalized_edit_similarity(left_sequence, right_sequence)
            )
    return float(1.0 - (weighted_similarity_sum / float(total_pairs)))


def compute_group_diversity_rewards(sequences, group_size):
    group_size = int(group_size)
    if group_size <= 1:
        raise ValueError(f'group_size must be greater than 1, got {group_size}')
    if len(sequences) % group_size != 0:
        raise ValueError(
            f'sequences length must be divisible by group_size: {len(sequences)} vs {group_size}'
        )
    rewards = []
    for start in range(0, len(sequences), group_size):
        valid_group = []
        for sequence in sequences[start:start + group_size]:
            if is_valid_protein_sequence(sequence):
                valid_group.append(normalize_protein_sequence(sequence))
        if len(valid_group) < 2:
            rewards.append(0.0)
            continue
        rewards.append(float(compute_group_diversity_reward(valid_group)))
    return rewards
