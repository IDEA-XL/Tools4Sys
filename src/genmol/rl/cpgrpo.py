from contextlib import nullcontext

import torch
import torch.nn.functional as F


def _cpu_generator(seed):
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    return generator


def sample_coupled_masks(completion_mask, seed, min_mask_ratio=0.2, max_mask_ratio=0.8):
    generator = _cpu_generator(seed)
    mask_ratio = min_mask_ratio + (max_mask_ratio - min_mask_ratio) * torch.rand((), generator=generator).item()
    random_matrix = torch.rand(completion_mask.shape, generator=generator)
    random_matrix = random_matrix.to(device=completion_mask.device, dtype=torch.float32)

    full_mask = completion_mask.clone()
    mask_a = completion_mask & (random_matrix < mask_ratio)
    mask_b = completion_mask & ~mask_a
    weights = torch.tensor(
        [1.0, 1.0 / mask_ratio, 1.0 / (1.0 - mask_ratio)],
        device=completion_mask.device,
        dtype=torch.float32,
    )
    return full_mask, mask_a, mask_b, weights


def apply_token_mask(token_ids, token_mask, mask_token_id):
    return torch.where(token_mask, torch.full_like(token_ids, mask_token_id), token_ids)


def selective_log_softmax(logits, target_ids, weights, mask_a, completion_mask):
    batch_size = target_ids.shape[0]
    log_probs = F.log_softmax(logits, dim=-1)

    full_log_probs = log_probs[:batch_size].gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    masked_log_probs = log_probs[batch_size:2 * batch_size].gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    reverse_masked_log_probs = log_probs[2 * batch_size:].gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

    weighted = torch.where(mask_a, masked_log_probs * weights[1], reverse_masked_log_probs * weights[2])
    final_log_probs = 0.5 * (full_log_probs + weighted)
    return torch.where(completion_mask, final_log_probs, torch.zeros_like(final_log_probs))


def compute_coupled_log_probs(score_fn, token_ids, completion_mask, mask_token_id, seeds, requires_grad):
    log_prob_batches = []
    mask_meta = []
    grad_context = nullcontext() if requires_grad else torch.no_grad()

    with grad_context:
        for seed in seeds:
            full_mask, mask_a, mask_b, weights = sample_coupled_masks(completion_mask, seed=seed)
            stacked_tokens = torch.cat(
                [
                    apply_token_mask(token_ids, full_mask, mask_token_id),
                    apply_token_mask(token_ids, mask_a, mask_token_id),
                    apply_token_mask(token_ids, mask_b, mask_token_id),
                ],
                dim=0,
            )
            logits = score_fn(stacked_tokens)
            log_prob_batches.append(
                selective_log_softmax(
                    logits=logits,
                    target_ids=token_ids,
                    weights=weights,
                    mask_a=mask_a,
                    completion_mask=completion_mask,
                )
            )
            mask_meta.append({'seed': int(seed), 'mask_ratio': float(1.0 / weights[1].item())})

    return torch.stack(log_prob_batches, dim=0), mask_meta


def compute_leave_one_out_advantages(rewards, group_size, scale_rewards=False):
    if group_size <= 1:
        raise ValueError('group_size must be greater than 1 for leave-one-out advantages')
    if rewards.numel() % group_size != 0:
        raise ValueError('rewards must be divisible by group_size')

    grouped = rewards.view(-1, group_size)
    baseline = (grouped.sum(dim=1, keepdim=True) - grouped) / (group_size - 1)
    advantages = (grouped - baseline).reshape(-1)

    if scale_rewards:
        std = grouped.std(dim=1, keepdim=True, unbiased=False).repeat_interleave(group_size, dim=1).reshape(-1)
        advantages = advantages / (std + 1e-4)

    return advantages


def masked_mean(values, mask):
    masked = values * mask
    denom = mask.sum().clamp(min=1.0)
    return masked.sum() / denom


def compute_clipped_grpo_loss(new_log_probs, old_log_probs, advantages, completion_mask, clip_range, ref_log_probs=None, beta=0.0):
    if new_log_probs.shape != old_log_probs.shape:
        raise ValueError('new_log_probs and old_log_probs must have the same shape')

    token_mask = completion_mask.unsqueeze(0).expand_as(new_log_probs).to(dtype=torch.float32)
    advantages = advantages.view(1, -1, 1)

    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    loss_unclipped = -advantages * ratio
    loss_clipped = -advantages * clipped_ratio
    per_token_loss = torch.maximum(loss_unclipped, loss_clipped)

    kl_mean = None
    if ref_log_probs is not None and beta > 0.0:
        per_token_kl = torch.exp(ref_log_probs - new_log_probs) - (ref_log_probs - new_log_probs) - 1.0
        per_token_loss = per_token_loss + beta * per_token_kl
        kl_mean = masked_mean(per_token_kl, token_mask).item()

    loss = masked_mean(per_token_loss, token_mask)

    sign_advantages = advantages.expand_as(ratio)
    is_low_clipped = (ratio < 1.0 - clip_range) & (sign_advantages < 0)
    is_high_clipped = (ratio > 1.0 + clip_range) & (sign_advantages > 0)
    clipped_region = (is_low_clipped | is_high_clipped).to(dtype=torch.float32)

    metrics = {
        'ratio_mean': masked_mean(ratio, token_mask).item(),
        'clip_ratio': masked_mean(clipped_region, token_mask).item(),
    }
    if kl_mean is not None:
        metrics['kl_mean'] = kl_mean

    return loss, metrics
