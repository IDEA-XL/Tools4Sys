import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from progen2.rewards.common import is_valid_protein_sequence, normalize_protein_sequence
from progen2.rewards.diversity import compute_group_diversity_reward, normalized_edit_similarity


_GLOBAL_EDIT_UNIQUE_SEQUENCES = None
_GLOBAL_EDIT_SEQUENCE_COUNTS = None


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


def _init_global_edit_diversity_worker(unique_sequences, sequence_counts):
    global _GLOBAL_EDIT_UNIQUE_SEQUENCES
    global _GLOBAL_EDIT_SEQUENCE_COUNTS
    _GLOBAL_EDIT_UNIQUE_SEQUENCES = tuple(unique_sequences)
    _GLOBAL_EDIT_SEQUENCE_COUNTS = dict(sequence_counts)


def _weighted_similarity_sum_for_range(index_range):
    if _GLOBAL_EDIT_UNIQUE_SEQUENCES is None or _GLOBAL_EDIT_SEQUENCE_COUNTS is None:
        raise RuntimeError('global edit diversity worker initialized without sequence state')
    start_index, end_index = index_range
    weighted_similarity_sum = 0.0
    unique_sequences = _GLOBAL_EDIT_UNIQUE_SEQUENCES
    sequence_counts = _GLOBAL_EDIT_SEQUENCE_COUNTS
    for left_idx in range(start_index, end_index):
        left_sequence = unique_sequences[left_idx]
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
    return weighted_similarity_sum


def global_edit_diversity_parallel(sequences, num_workers=None):
    valid_sequences = [normalize_protein_sequence(sequence) for sequence in sequences if is_valid_protein_sequence(sequence)]
    if len(valid_sequences) < 2:
        return 0.0
    sequence_counts = Counter(valid_sequences)
    unique_sequences = list(sequence_counts.keys())
    total_pairs = len(valid_sequences) * (len(valid_sequences) - 1) // 2
    if total_pairs <= 0:
        return 0.0
    if num_workers is None:
        num_workers = os.cpu_count() or 1
    num_workers = int(num_workers)
    if num_workers <= 1 or len(unique_sequences) < 2:
        return global_edit_diversity(valid_sequences)
    worker_count = min(num_workers, len(unique_sequences))
    chunk_size = math.ceil(len(unique_sequences) / worker_count)
    index_ranges = [
        (start_index, min(start_index + chunk_size, len(unique_sequences)))
        for start_index in range(0, len(unique_sequences), chunk_size)
    ]
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_global_edit_diversity_worker,
        initargs=(unique_sequences, sequence_counts),
    ) as executor:
        weighted_similarity_sum = sum(executor.map(_weighted_similarity_sum_for_range, index_ranges))
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
