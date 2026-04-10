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
                completion_mask=completion_mask,
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
        per_token_logps = selective_log_softmax(
            logits=logits,
            index=expanded_input,
            weights=weights,
            mask=partial_mask,
        ).view(num_iterations, batch_size, seq_len).permute(1, 0, 2)

    return per_token_logps.to(torch.float32)
