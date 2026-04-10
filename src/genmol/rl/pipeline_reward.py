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
