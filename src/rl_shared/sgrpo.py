import math

import torch

VALID_SGRPO_HIERARCHIES = {'advantage_sum', 'reward_sum'}


def normalize_reward_thresholds(thresholds):
    if thresholds is None:
        return {}
    if not isinstance(thresholds, dict):
        raise ValueError(
            'individual_reward_thresholds must be a dict keyed by reward name or null'
        )

    normalized = {}
    for reward_name, value in thresholds.items():
        if not isinstance(reward_name, str) or not reward_name:
            raise ValueError(
                f'individual_reward_thresholds keys must be non-empty strings, got {reward_name!r}'
            )
        if value is None:
            normalized[reward_name] = None
            continue
        try:
            normalized[reward_name] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'individual_reward_thresholds[{reward_name!r}] must be a float or null, got {value!r}'
            ) from exc
    return normalized


def validate_reward_threshold_names(thresholds, available_reward_names):
    normalized = normalize_reward_thresholds(thresholds)
    unknown = sorted(set(normalized) - set(available_reward_names))
    if unknown:
        raise ValueError(
            'individual_reward_thresholds contains unsupported reward names: '
            f'{unknown}; expected subset of {sorted(available_reward_names)}'
        )
    return normalized


def _compute_group_reward_indicator(group_mean_individual_rewards, group_reward_thresholds, num_groups, device):
    indicator = torch.ones(num_groups, device=device, dtype=torch.float32)
    active_threshold_count = 0
    for reward_name, threshold in group_reward_thresholds.items():
        if threshold is None:
            continue
        if reward_name not in group_mean_individual_rewards:
            raise ValueError(
                f'missing group mean individual reward tensor for thresholded reward {reward_name!r}'
            )
        reward_means = group_mean_individual_rewards[reward_name]
        if reward_means.dim() != 1:
            raise ValueError(f'group mean individual rewards for {reward_name!r} must be 1D')
        if reward_means.numel() != num_groups:
            raise ValueError(
                'group mean individual reward tensor length must match number of groups: '
                f'{reward_name!r} has {reward_means.numel()} vs {num_groups}'
            )
        indicator = indicator * (reward_means > float(threshold)).to(dtype=torch.float32)
        active_threshold_count += 1
    return indicator, active_threshold_count


def compute_grouped_advantages(rewards, num_generations, scale_rewards=False):
    if rewards.dim() != 1:
        raise ValueError('rewards must be a 1D tensor')
    if rewards.numel() % num_generations != 0:
        raise ValueError('rewards length must be divisible by num_generations')

    rewards_grouped = rewards.view(-1, num_generations)
    sum_group = rewards_grouped.sum(dim=1, keepdim=True)
    baseline = (sum_group - rewards_grouped) / (num_generations - 1)
    advantages = (rewards_grouped - baseline).view(-1)
    std_grouped = rewards_grouped.std(dim=1, keepdim=True)
    repeated_std = std_grouped.repeat_interleave(num_generations, dim=1).view(-1)
    if scale_rewards:
        advantages = advantages / (repeated_std + 1e-4)
    zero_std_ratio = (repeated_std < 1e-6).to(torch.float32).mean().item()
    return advantages, repeated_std, zero_std_ratio


def compute_sgrpo_advantages(
    rollout_rewards,
    group_rewards,
    num_generations,
    supergroup_num_groups,
    group_advantage_weight,
    scale_rewards=False,
    hierarchy='advantage_sum',
    group_mean_individual_rewards=None,
    individual_reward_thresholds=None,
):
    if rollout_rewards.dim() != 1:
        raise ValueError('rollout_rewards must be a 1D tensor')
    if group_rewards.dim() != 1:
        raise ValueError('group_rewards must be a 1D tensor')
    if num_generations <= 1:
        raise ValueError('num_generations must be greater than 1')
    if supergroup_num_groups <= 1:
        raise ValueError('supergroup_num_groups must be greater than 1')
    if not 0.0 <= group_advantage_weight <= 1.0:
        raise ValueError('group_advantage_weight must be in [0, 1]')
    if hierarchy not in VALID_SGRPO_HIERARCHIES:
        raise ValueError(
            f"hierarchy must be one of {sorted(VALID_SGRPO_HIERARCHIES)}, got {hierarchy!r}"
        )
    if rollout_rewards.numel() % num_generations != 0:
        raise ValueError('rollout_rewards length must be divisible by num_generations')

    num_groups = rollout_rewards.numel() // num_generations
    if group_rewards.numel() != num_groups:
        raise ValueError(
            f'group_rewards length must equal number of groups: {group_rewards.numel()} vs {num_groups}'
        )
    if num_groups % supergroup_num_groups != 0:
        raise ValueError(
            'number of groups must be divisible by supergroup_num_groups: '
            f'{num_groups} vs {supergroup_num_groups}'
        )

    group_mean_individual_rewards = (
        {} if group_mean_individual_rewards is None else dict(group_mean_individual_rewards)
    )
    group_reward_thresholds = normalize_reward_thresholds(individual_reward_thresholds)
    group_reward_indicator, active_threshold_count = _compute_group_reward_indicator(
        group_mean_individual_rewards=group_mean_individual_rewards,
        group_reward_thresholds=group_reward_thresholds,
        num_groups=num_groups,
        device=group_rewards.device,
    )
    gated_group_rewards = group_rewards * group_reward_indicator

    rollout_supergroup_size = num_generations * supergroup_num_groups
    rollout_advantages, rollout_reward_std, rollout_zero_std_ratio = compute_grouped_advantages(
        rewards=rollout_rewards,
        num_generations=rollout_supergroup_size,
        scale_rewards=scale_rewards,
    )
    group_advantages, group_reward_std, group_zero_std_ratio = compute_grouped_advantages(
        rewards=gated_group_rewards,
        num_generations=supergroup_num_groups,
        scale_rewards=scale_rewards,
    )
    expanded_group_advantages = group_advantages.repeat_interleave(num_generations)
    rollout_advantage_weight = 1.0 - group_advantage_weight
    if hierarchy == 'advantage_sum':
        final_advantages = (
            rollout_advantages * rollout_advantage_weight
            + expanded_group_advantages * group_advantage_weight
        )
        final_reward_std = rollout_reward_std
        final_zero_std_ratio = rollout_zero_std_ratio
        combined_reward_mean = float('nan')
    else:
        expanded_group_rewards = gated_group_rewards.repeat_interleave(num_generations)
        combined_rewards = (
            rollout_rewards * rollout_advantage_weight
            + expanded_group_rewards * group_advantage_weight
        )
        final_advantages, final_reward_std, final_zero_std_ratio = compute_grouped_advantages(
            rewards=combined_rewards,
            num_generations=rollout_supergroup_size,
            scale_rewards=scale_rewards,
        )
        combined_reward_mean = combined_rewards.mean().item()

    metrics = {
        'rollout_advantage_mean': rollout_advantages.mean().item(),
        'group_advantage_mean': expanded_group_advantages.mean().item(),
        'group_reward_mean': gated_group_rewards.mean().item(),
        'group_reward_raw_mean': group_rewards.mean().item(),
        'group_advantage_weight': float(group_advantage_weight),
        'rollout_zero_std_ratio': final_zero_std_ratio,
        'group_zero_std_ratio': group_zero_std_ratio,
        'rollout_reward_std_mean': final_reward_std.mean().item(),
        'group_reward_std_mean': group_reward_std.mean().item(),
        'group_reward_indicator_mean': group_reward_indicator.mean().item(),
        'active_threshold_count': float(active_threshold_count),
        'hierarchy_reward_sum_enabled': 1.0 if hierarchy == 'reward_sum' else 0.0,
        'combined_reward_mean': combined_reward_mean,
    }
    return final_advantages, expanded_group_advantages, rollout_advantages, metrics


def compute_group_reward_regularizer_advantages(group_rewards, num_generations, scale_rewards=False):
    if group_rewards.dim() != 1:
        raise ValueError('group_rewards must be a 1D tensor')
    if num_generations <= 0:
        raise ValueError('num_generations must be positive')

    num_groups = int(group_rewards.numel())
    if num_groups <= 1:
        raise ValueError('group_rewards must contain at least two groups for a centered regularizer signal')

    group_advantages, group_reward_std, group_zero_std_ratio = compute_grouped_advantages(
        rewards=group_rewards,
        num_generations=num_groups,
        scale_rewards=scale_rewards,
    )
    expanded_group_advantages = group_advantages.repeat_interleave(num_generations)
    metrics = {
        'group_advantage_mean': expanded_group_advantages.mean().item(),
        'group_reward_mean': group_rewards.mean().item(),
        'group_zero_std_ratio': group_zero_std_ratio,
        'group_reward_std_mean': group_reward_std.mean().item(),
    }
    return expanded_group_advantages, metrics


def compute_clipped_grpo_loss(
    new_log_probs,
    old_log_probs,
    advantages,
    completion_mask,
    epsilon,
    ref_log_probs=None,
    beta=0.0,
):
    if new_log_probs.shape != old_log_probs.shape:
        raise ValueError('new_log_probs and old_log_probs must have the same shape')
    if new_log_probs.dim() != 3 or new_log_probs.shape[1] != 1:
        raise ValueError('expected log prob tensors to have shape [batch, 1, seq_len]')
    if completion_mask.dim() != 2:
        raise ValueError('completion_mask must have shape [batch, seq_len]')

    completion_mask = completion_mask.to(dtype=torch.float32)
    advantages = advantages.view(-1, 1, 1)
    coef_1 = torch.exp(new_log_probs - old_log_probs)
    coef_2 = torch.clamp(coef_1, 1 - epsilon, 1 + epsilon)
    loss_1 = coef_1 * advantages
    loss_2 = coef_2 * advantages
    per_token_loss = -torch.min(loss_1, loss_2)

    kl_value = None
    if ref_log_probs is not None and beta != 0.0:
        per_token_kl = torch.exp(ref_log_probs - new_log_probs) - (ref_log_probs - new_log_probs) - 1.0
        per_token_loss = per_token_loss + beta * per_token_kl
        kl_value = (per_token_kl[:, 0, :] * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)

    loss = (per_token_loss[:, 0, :] * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)

    is_low_clipped = (coef_1 < 1 - epsilon) & (advantages < 0)
    is_high_clipped = (coef_1 > 1 + epsilon) & (advantages > 0)
    is_region_clipped = is_low_clipped | is_high_clipped

    low_clip = (is_low_clipped[:, 0, :].to(torch.float32) * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
    high_clip = (is_high_clipped[:, 0, :].to(torch.float32) * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
    region_clip = (is_region_clipped[:, 0, :].to(torch.float32) * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
    ratio_mean = (coef_1[:, 0, :] * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)

    metrics = {
        'ratio_mean': ratio_mean.detach(),
        'clip_ratio_low_mean': low_clip.detach(),
        'clip_ratio_high_mean': high_clip.detach(),
        'clip_ratio_region_mean': region_clip.detach(),
    }
    if kl_value is not None:
        metrics['kl_mean'] = kl_value.detach()

    return loss, metrics


def compute_warmup_steps(max_steps, warmup_ratio):
    return max(1, int(math.ceil(max_steps * warmup_ratio)))
