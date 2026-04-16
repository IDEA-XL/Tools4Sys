import math

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
    similarities = []
    for left_idx in range(len(valid_sequences)):
        for right_idx in range(left_idx + 1, len(valid_sequences)):
            similarities.append(normalized_edit_similarity(valid_sequences[left_idx], valid_sequences[right_idx]))
    if not similarities:
        return 0.0
    return float(1.0 - (sum(similarities) / len(similarities)))


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
