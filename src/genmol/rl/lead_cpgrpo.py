from contextlib import nullcontext

import torch
import torch.nn.functional as F

from genmol.rl.cpgrpo import forward_process


def _selective_log_softmax_materialized(logits, index, weights, mask):
    full_batch_size = logits.size(0) // 3
    if full_batch_size == 0:
        raise ValueError('logits batch size must be at least 3')
    if weights is None or mask is None:
        raise ValueError('weights and mask are required for coupled selective_log_softmax')
    if logits.size(0) % 3 != 0:
        raise ValueError('logits batch size must be divisible by 3')

    num_iterations = weights.size(0) // 3
    if num_iterations == 0 or weights.size(0) % 3 != 0:
        raise ValueError('weights length must be a positive multiple of 3')
    if full_batch_size % num_iterations != 0:
        raise ValueError('full batch size must be divisible by num_iterations')
    if index.size(0) != full_batch_size or mask.size(0) != full_batch_size:
        raise ValueError('index and mask must align with full batch size')
    if index.shape != mask.shape:
        raise ValueError('index and mask must have the same shape')

    batch_size = full_batch_size // num_iterations
    seq_len = index.size(1)
    vocab_size = logits.size(-1)

    logits_reshaped = logits.reshape(num_iterations, 3, batch_size, seq_len, vocab_size)
    logps = F.log_softmax(logits_reshaped, dim=-1)

    index_reshaped = index.reshape(num_iterations, batch_size, seq_len).clone()
    mask_reshaped = mask.reshape(num_iterations, batch_size, seq_len).clone()
    gather_index = (
        torch.stack([index_reshaped, index_reshaped, index_reshaped], dim=1)
        .unsqueeze(-1)
        .contiguous()
    )
    gathered = logps.gather(dim=-1, index=gather_index).squeeze(-1)

    weight_tensor = weights.reshape(num_iterations, 3, 1, 1)
    weighted = torch.where(
        mask_reshaped,
        gathered[:, 1] * weight_tensor[:, 1],
        gathered[:, 2] * weight_tensor[:, 2],
    )
    return ((gathered[:, 0] + weighted) / 2).reshape(full_batch_size, seq_len)


def get_per_token_logps_full(
    score_fn,
    input_ids,
    completion_mask,
    mask_token_id,
    mask_seeds,
    gradient_accumulation_steps,
    requires_grad,
    score_chunk_size=None,
):
    if input_ids.dim() != 3:
        raise ValueError(f'Expected input_ids to have 3 dimensions, got {input_ids.dim()}')
    if completion_mask.dim() != 2:
        raise ValueError(f'Expected completion_mask to have 2 dimensions, got {completion_mask.dim()}')

    input_ids = input_ids.clone()
    completion_mask = completion_mask.clone()

    num_iterations, batch_size, seq_len = input_ids.size()
    if completion_mask.size(0) != batch_size or completion_mask.size(1) != seq_len:
        raise ValueError(
            'completion_mask must have shape '
            f'[{batch_size}, {seq_len}], got {list(completion_mask.shape)}'
        )
    if len(mask_seeds) != num_iterations:
        raise ValueError(f'Expected {num_iterations} mask seeds, got {len(mask_seeds)}')
    if score_chunk_size is not None and score_chunk_size <= 0:
        raise ValueError('score_chunk_size must be positive when provided')

    grad_context = nullcontext() if requires_grad else torch.no_grad()
    with grad_context:
        chunk_size = batch_size if score_chunk_size is None else min(score_chunk_size, batch_size)
        iteration_outputs = []

        for iteration_idx, mask_seed in enumerate(mask_seeds):
            expanded_input = input_ids[iteration_idx].clone()
            perturbed, weights, partial_mask = forward_process(
                batch=expanded_input,
                completion_mask=completion_mask,
                mask_id=mask_token_id,
                seed=mask_seed,
                gradient_accumulation_steps=gradient_accumulation_steps,
                accumulate=num_iterations > 1,
            )
            perturbed = [item.clone() for item in perturbed]
            partial_mask = partial_mask.clone()
            weight_tensor = torch.tensor(weights, device=input_ids.device, dtype=torch.float32)
            chunk_outputs = []
            for start in range(0, batch_size, chunk_size):
                end = min(start + chunk_size, batch_size)
                perturbed_seq = torch.cat([item[start:end] for item in perturbed], dim=0)
                logits = score_fn(perturbed_seq)
                chunk_targets = expanded_input[start:end].clone()
                chunk_mask = partial_mask[start:end].clone()
                chunk_outputs.append(
                    _selective_log_softmax_materialized(
                        logits=logits,
                        index=chunk_targets,
                        weights=weight_tensor,
                        mask=chunk_mask,
                    )
                )
            iteration_outputs.append(torch.cat(chunk_outputs, dim=0))

        per_token_logps = torch.stack(iteration_outputs, dim=1)

    return per_token_logps.to(torch.float32)
