import random

import torch


def topk_mean(values, k):
    if values.dim() != 1:
        raise ValueError(f'Expected a 1D tensor for topk_mean, got shape {list(values.shape)}')
    if values.numel() == 0:
        raise ValueError('topk_mean requires at least one value')
    if k <= 0:
        raise ValueError(f'topk must be positive, got {k}')
    effective_k = min(int(k), int(values.numel()))
    return torch.topk(values, k=effective_k).values.mean()


def combine_seed_rewards(seed_base_rewards, downstream_base_rewards, alpha):
    if seed_base_rewards.shape != downstream_base_rewards.shape:
        raise ValueError(
            'seed_base_rewards and downstream_base_rewards must have the same shape: '
            f'{list(seed_base_rewards.shape)} vs {list(downstream_base_rewards.shape)}'
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f'alpha must be in [0, 1], got {alpha}')
    return (1.0 - alpha) * seed_base_rewards + alpha * downstream_base_rewards


def lead_base_reward_from_total(total_reward, similarity, sim_weight, is_valid):
    if not is_valid:
        return -1.0
    if similarity is None:
        raise ValueError('similarity must be present for valid lead rewards')
    return float(total_reward) - float(sim_weight) * float(similarity)


def select_full_denovo_groups_for_lead(
    seed_entries,
    denovo_group_size,
    lead_num_seed_groups,
    seed,
):
    if denovo_group_size <= 0:
        raise ValueError(f'denovo_group_size must be positive, got {denovo_group_size}')
    if lead_num_seed_groups <= 0:
        raise ValueError(f'lead_num_seed_groups must be positive, got {lead_num_seed_groups}')
    if len(seed_entries) % denovo_group_size != 0:
        raise ValueError(
            'seed_entries length must be divisible by denovo_group_size: '
            f'{len(seed_entries)} vs {denovo_group_size}'
        )
    if lead_num_seed_groups % denovo_group_size != 0:
        raise ValueError(
            'lead_num_seed_groups must be divisible by denovo_group_size so full de novo groups can be selected: '
            f'{lead_num_seed_groups} vs {denovo_group_size}'
        )

    num_denovo_groups = len(seed_entries) // denovo_group_size
    required_denovo_groups = lead_num_seed_groups // denovo_group_size

    candidate_group_indices = []
    for group_idx in range(num_denovo_groups):
        start = group_idx * denovo_group_size
        end = start + denovo_group_size
        group_entries = seed_entries[start:end]
        if all(bool(item['is_valid']) and item.get('smiles') for item in group_entries):
            candidate_group_indices.append(group_idx)

    if len(candidate_group_indices) < required_denovo_groups:
        raise ValueError(
            'Not enough fully valid de novo groups to satisfy lead seed sampling: '
            f'{len(candidate_group_indices)} available vs {required_denovo_groups} required'
        )

    rng = random.Random(int(seed))
    selected_group_indices = sorted(rng.sample(candidate_group_indices, required_denovo_groups))
    selected_seed_entries = []
    selected_seed_mask = [False] * len(seed_entries)
    for group_idx in selected_group_indices:
        start = group_idx * denovo_group_size
        end = start + denovo_group_size
        for global_seed_index in range(start, end):
            entry = seed_entries[global_seed_index]
            selected_seed_entries.append(
                {
                    'global_seed_index': global_seed_index,
                    'smiles': entry['smiles'],
                }
            )
            selected_seed_mask[global_seed_index] = True

    if len(selected_seed_entries) != lead_num_seed_groups:
        raise ValueError(
            'Selected seed count does not match requested lead_num_seed_groups: '
            f'{len(selected_seed_entries)} vs {lead_num_seed_groups}'
        )

    return selected_group_indices, selected_seed_entries, selected_seed_mask


def partition_selected_seed_entries(selected_seed_entries, num_partitions):
    if num_partitions <= 0:
        raise ValueError(f'num_partitions must be positive, got {num_partitions}')
    if len(selected_seed_entries) % num_partitions != 0:
        raise ValueError(
            'selected_seed_entries length must be divisible by num_partitions: '
            f'{len(selected_seed_entries)} vs {num_partitions}'
        )
    shard_size = len(selected_seed_entries) // num_partitions
    return [
        selected_seed_entries[partition_idx * shard_size:(partition_idx + 1) * shard_size]
        for partition_idx in range(num_partitions)
    ]


def aggregate_selected_seed_downstream_base_rewards(
    selected_seed_entries,
    lead_base_rewards,
    lead_num_generations,
    downstream_topk,
):
    if lead_num_generations <= 0:
        raise ValueError(f'lead_num_generations must be positive, got {lead_num_generations}')
    if lead_base_rewards.dim() != 1:
        raise ValueError(
            f'Expected lead_base_rewards to be 1D, got shape {list(lead_base_rewards.shape)}'
        )
    expected = len(selected_seed_entries) * lead_num_generations
    if lead_base_rewards.numel() != expected:
        raise ValueError(
            'lead_base_rewards size does not match selected seed layout: '
            f'{lead_base_rewards.numel()} vs {expected}'
        )

    aggregated = []
    for seed_idx, seed_entry in enumerate(selected_seed_entries):
        start = seed_idx * lead_num_generations
        end = start + lead_num_generations
        aggregated.append(
            {
                'global_seed_index': int(seed_entry['global_seed_index']),
                'downstream_base_reward': float(topk_mean(lead_base_rewards[start:end], k=downstream_topk)),
            }
        )
    return aggregated


def merge_selected_seed_downstream_base_rewards(
    default_downstream_base_rewards,
    selected_seed_rewards,
    expected_selected_seed_count,
):
    merged = [float(value) for value in default_downstream_base_rewards]
    if expected_selected_seed_count < 0:
        raise ValueError(
            f'expected_selected_seed_count must be non-negative, got {expected_selected_seed_count}'
        )

    seen_indices = set()
    for item in selected_seed_rewards:
        global_seed_index = int(item['global_seed_index'])
        if global_seed_index < 0 or global_seed_index >= len(merged):
            raise ValueError(
                'global_seed_index out of bounds when merging downstream base rewards: '
                f'{global_seed_index} vs {len(merged)}'
            )
        if global_seed_index in seen_indices:
            raise ValueError(
                f'duplicate global_seed_index in selected_seed_rewards: {global_seed_index}'
            )
        seen_indices.add(global_seed_index)
        merged[global_seed_index] = float(item['downstream_base_reward'])

    if len(seen_indices) != expected_selected_seed_count:
        raise ValueError(
            'selected_seed_rewards count mismatch when merging downstream base rewards: '
            f'{len(seen_indices)} vs {expected_selected_seed_count}'
        )

    return merged
