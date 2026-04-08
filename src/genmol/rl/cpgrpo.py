import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F


def _cpu_generator(seed):
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    return generator


def split_tensor_dict(tensor_dict, num_chunks):
    first_tensor = next(value for value in tensor_dict.values() if isinstance(value, torch.Tensor))
    if first_tensor.shape[0] % num_chunks != 0:
        raise ValueError('tensor batch dimension must be divisible by num_chunks')

    chunk_size = first_tensor.shape[0] // num_chunks
    chunks = []
    for chunk_idx in range(num_chunks):
        chunk = {}
        start = chunk_idx * chunk_size
        end = (chunk_idx + 1) * chunk_size
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor) and value.shape[:1] == first_tensor.shape[:1]:
                chunk[key] = value[start:end]
            else:
                chunk[key] = value
        chunks.append(chunk)
    return chunks


def selective_log_softmax(logits, index, weights=None, mask=None):
    full_batch_size = logits.size(0) // 3
    if full_batch_size == 0:
        raise ValueError('logits batch size must be at least 3')

    if weights is None or mask is None:
        raise ValueError('weights and mask are required for coupled selective_log_softmax')

    num_iterations = weights.size(0) // 3
    batch_size = full_batch_size // num_iterations
    per_token_logps = []

    for sample_idx in range(full_batch_size):
        labels = index[sample_idx]
        chunk_idx, offset = divmod(sample_idx, batch_size)
        base = chunk_idx * 3 * batch_size
        logits_index = torch.tensor(
            [base + offset, base + batch_size + offset, base + 2 * batch_size + offset],
            device=logits.device,
        )
        seq_logits = logits[logits_index]
        seq_logps = F.log_softmax(seq_logits, dim=-1)
        gathered = seq_logps.gather(
            dim=-1,
            index=labels.unsqueeze(0).unsqueeze(-1).expand(3, -1, 1),
        ).squeeze(-1)

        seq_weights = weights[chunk_idx * 3:(chunk_idx + 1) * 3]
        seq_mask = mask[sample_idx]
        weighted = torch.where(seq_mask, gathered[1] * seq_weights[1], gathered[2] * seq_weights[2])
        per_token_logps.append((gathered[0] + weighted) / 2)

    return torch.stack(per_token_logps, dim=0)


def forward_process(
    batch,
    completion_mask,
    mask_id,
    seed,
    gradient_accumulation_steps=1,
    accumulate=False,
):
    generator = _cpu_generator(seed)
    batch_size, seq_len = batch.shape
    mask_ratio = 0.2 + 0.6 * torch.rand((), generator=generator).item()

    if accumulate:
        if batch_size % gradient_accumulation_steps != 0:
            raise ValueError('batch size must be divisible by gradient_accumulation_steps when accumulate=True')
        random_matrix = torch.rand((batch_size // gradient_accumulation_steps, seq_len), generator=generator)
        random_matrix = torch.cat([random_matrix] * gradient_accumulation_steps, dim=0)
    else:
        random_matrix = torch.rand((batch_size, seq_len), generator=generator)
    random_matrix = random_matrix.to(device=batch.device, dtype=torch.float32)

    full_mask = completion_mask
    mask_a = completion_mask & (random_matrix < mask_ratio)
    mask_b = completion_mask & (random_matrix > mask_ratio)
    noisy_batch = [
        torch.where(full_mask, mask_id, batch),
        torch.where(mask_a, mask_id, batch),
        torch.where(mask_b, mask_id, batch),
    ]
    return noisy_batch, [1.0, 1.0 / mask_ratio, 1.0 / (1.0 - mask_ratio)], mask_a


def get_per_token_logps(
    score_fn,
    input_ids,
    logits_to_keep,
    completion_mask,
    mask_token_id,
    mask_seeds,
    gradient_accumulation_steps,
    requires_grad,
):
    if input_ids.dim() != 3:
        raise ValueError(f'Expected input_ids to have 3 dimensions, got {input_ids.dim()}')
    if completion_mask.dim() != 2:
        raise ValueError(f'Expected completion_mask to have 2 dimensions, got {completion_mask.dim()}')

    num_iterations, batch_size, seq_len = input_ids.size()
    if logits_to_keep <= 0 or logits_to_keep > seq_len:
        raise ValueError(f'logits_to_keep must be in [1, {seq_len}], got {logits_to_keep}')
    if completion_mask.size(0) != batch_size or completion_mask.size(1) != logits_to_keep:
        raise ValueError(
            'completion_mask must have shape '
            f'[{batch_size}, {logits_to_keep}], got {list(completion_mask.shape)}'
        )
    if len(mask_seeds) != num_iterations:
        raise ValueError(f'Expected {num_iterations} mask seeds, got {len(mask_seeds)}')

    prompt_length = seq_len - logits_to_keep
    full_completion_mask = torch.zeros((batch_size, seq_len), device=input_ids.device, dtype=torch.bool)
    full_completion_mask[:, prompt_length:] = completion_mask

    grad_context = nullcontext() if requires_grad else torch.no_grad()
    with grad_context:
        all_perturbed = []
        all_weights = []
        all_expanded_inputs = []
        all_partial_masks = []

        for iteration_idx, mask_seed in enumerate(mask_seeds):
            expanded_input = input_ids[iteration_idx]
            perturbed, weights, partial_mask = forward_process(
                batch=expanded_input,
                completion_mask=full_completion_mask,
                mask_id=mask_token_id,
                seed=mask_seed,
                gradient_accumulation_steps=gradient_accumulation_steps,
                accumulate=num_iterations > 1,
            )
            all_perturbed.extend(perturbed)
            all_weights.extend(weights)
            all_expanded_inputs.append(expanded_input)
            all_partial_masks.append(partial_mask)

        perturbed_seq = torch.cat(all_perturbed, dim=0)
        expanded_input = torch.cat(all_expanded_inputs, dim=0)
        partial_mask = torch.cat(all_partial_masks, dim=0)
        weights = torch.tensor(all_weights, device=input_ids.device, dtype=torch.float32)

        logits = score_fn(perturbed_seq)
        completion_logits = logits[:, -logits_to_keep:, :]
        completion_targets = expanded_input[:, -logits_to_keep:]
        completion_loss_mask = partial_mask[:, -logits_to_keep:]
        per_token_logps = selective_log_softmax(
            logits=completion_logits,
            index=completion_targets,
            weights=weights,
            mask=completion_loss_mask,
        ).view(num_iterations, batch_size, logits_to_keep).permute(1, 0, 2)

    return per_token_logps.to(torch.float32)


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
