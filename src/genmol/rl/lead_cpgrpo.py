from contextlib import nullcontext

import torch

from genmol.rl.cpgrpo import forward_process, selective_log_softmax


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
            expanded_input = input_ids[iteration_idx]
            perturbed, weights, partial_mask = forward_process(
                batch=expanded_input,
                completion_mask=completion_mask,
                mask_id=mask_token_id,
                seed=mask_seed,
                gradient_accumulation_steps=gradient_accumulation_steps,
                accumulate=num_iterations > 1,
            )
            weight_tensor = torch.tensor(weights, device=input_ids.device, dtype=torch.float32)
            chunk_outputs = []
            for start in range(0, batch_size, chunk_size):
                end = min(start + chunk_size, batch_size)
                perturbed_seq = torch.cat([item[start:end] for item in perturbed], dim=0)
                logits = score_fn(perturbed_seq)
                chunk_outputs.append(
                    selective_log_softmax(
                        logits=logits,
                        index=expanded_input[start:end],
                        weights=weight_tensor,
                        mask=partial_mask[start:end],
                    )
                )
            iteration_outputs.append(torch.cat(chunk_outputs, dim=0))

        per_token_logps = torch.stack(iteration_outputs, dim=1)

    return per_token_logps.to(torch.float32)
