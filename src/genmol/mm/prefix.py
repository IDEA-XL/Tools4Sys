from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PackedPrefixBatch:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    token_positions: torch.Tensor
    prefix_lengths: torch.Tensor


def pad_prefix_embeddings(prefix_embeddings, device=None, dtype=None):
    _validate_prefix_list(prefix_embeddings)
    target_device = device or prefix_embeddings[0].device
    target_dtype = dtype or prefix_embeddings[0].dtype
    batch_size = len(prefix_embeddings)
    hidden_size = prefix_embeddings[0].size(1)
    max_prefix_len = max(int(embedding.size(0)) for embedding in prefix_embeddings)
    padded = torch.zeros(
        (batch_size, max_prefix_len, hidden_size),
        device=target_device,
        dtype=target_dtype,
    )
    mask = torch.zeros((batch_size, max_prefix_len), device=target_device, dtype=torch.bool)
    for batch_idx, embedding in enumerate(prefix_embeddings):
        prefix_len = int(embedding.size(0))
        padded[batch_idx, :prefix_len] = embedding.to(device=target_device, dtype=target_dtype)
        mask[batch_idx, :prefix_len] = True
    return padded, mask


def _validate_prefix_list(prefix_embeddings):
    if not prefix_embeddings:
        raise ValueError('prefix_embeddings must be non-empty')
    first_dim = None
    for idx, embedding in enumerate(prefix_embeddings):
        if not torch.is_tensor(embedding):
            raise TypeError(f'prefix embedding at index {idx} must be a tensor')
        if embedding.dim() != 2:
            raise ValueError(
                f'prefix embedding at index {idx} must have shape [prefix_len, hidden_size], '
                f'got {list(embedding.shape)}'
            )
        if first_dim is None:
            first_dim = embedding.size(1)
        elif embedding.size(1) != first_dim:
            raise ValueError(
                'all prefix embeddings must share the same hidden size: '
                f'{embedding.size(1)} vs {first_dim}'
            )


def pack_prefix_conditioning(token_embeddings, token_attention_mask, prefix_embeddings, max_total_positions):
    if token_embeddings.dim() != 3:
        raise ValueError('token_embeddings must have shape [batch, seq_len, hidden_size]')
    if token_attention_mask.dim() != 2:
        raise ValueError('token_attention_mask must have shape [batch, seq_len]')
    if token_embeddings.size(0) != token_attention_mask.size(0):
        raise ValueError('token_embeddings and token_attention_mask batch sizes must match')
    if token_embeddings.size(1) != token_attention_mask.size(1):
        raise ValueError('token_embeddings and token_attention_mask sequence lengths must match')
    if max_total_positions <= 0:
        raise ValueError(f'max_total_positions must be positive, got {max_total_positions}')

    _validate_prefix_list(prefix_embeddings)
    batch_size, seq_len, hidden_size = token_embeddings.shape
    if len(prefix_embeddings) != batch_size:
        raise ValueError(
            f'Expected {batch_size} prefix embeddings, got {len(prefix_embeddings)}'
        )

    device = token_embeddings.device
    dtype = token_embeddings.dtype
    prefix_lengths = []
    total_lengths = []
    token_lengths = token_attention_mask.to(dtype=torch.long).sum(dim=1)
    for batch_idx, prefix in enumerate(prefix_embeddings):
        if prefix.size(1) != hidden_size:
            raise ValueError(
                f'Prefix hidden size mismatch at batch index {batch_idx}: '
                f'{prefix.size(1)} vs {hidden_size}'
            )
        prefix_len = int(prefix.size(0))
        token_len = int(token_lengths[batch_idx].item())
        total_len = prefix_len + token_len
        if total_len > max_total_positions:
            raise ValueError(
                'Sample exceeds max_total_positions: '
                f'prefix_len={prefix_len} token_len={token_len} '
                f'total_len={total_len} max_total_positions={max_total_positions}'
            )
        prefix_lengths.append(prefix_len)
        total_lengths.append(total_len)

    max_total_len = max(total_lengths)
    inputs_embeds = torch.zeros(
        (batch_size, max_total_len, hidden_size),
        device=device,
        dtype=dtype,
    )
    attention_mask = torch.zeros((batch_size, max_total_len), device=device, dtype=torch.bool)
    position_ids = torch.zeros((batch_size, max_total_len), device=device, dtype=torch.long)
    token_positions = torch.full((batch_size, seq_len), -1, device=device, dtype=torch.long)

    for batch_idx in range(batch_size):
        prefix = prefix_embeddings[batch_idx].to(device=device, dtype=dtype)
        prefix_len = prefix_lengths[batch_idx]
        token_len = int(token_lengths[batch_idx].item())
        total_len = total_lengths[batch_idx]
        if prefix_len > 0:
            inputs_embeds[batch_idx, :prefix_len] = prefix
        if token_len > 0:
            inputs_embeds[batch_idx, prefix_len:total_len] = token_embeddings[batch_idx, :token_len]
            token_positions[batch_idx, :token_len] = torch.arange(
                prefix_len,
                total_len,
                device=device,
                dtype=torch.long,
            )
        attention_mask[batch_idx, :total_len] = True
        position_ids[batch_idx, :total_len] = torch.arange(total_len, device=device, dtype=torch.long)

    return PackedPrefixBatch(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        token_positions=token_positions,
        prefix_lengths=torch.tensor(prefix_lengths, device=device, dtype=torch.long),
    )


def extract_molecule_logits(logits, token_positions):
    if logits.dim() != 3:
        raise ValueError('logits must have shape [batch, seq_len, vocab_size]')
    if token_positions.dim() != 2:
        raise ValueError('token_positions must have shape [batch, token_seq_len]')
    if logits.size(0) != token_positions.size(0):
        raise ValueError('logits and token_positions batch sizes must match')

    batch_size, token_seq_len = token_positions.shape
    vocab_size = logits.size(-1)
    gathered = torch.zeros(
        (batch_size, token_seq_len, vocab_size),
        device=logits.device,
        dtype=logits.dtype,
    )
    for batch_idx in range(batch_size):
        positions = token_positions[batch_idx]
        valid = positions >= 0
        if valid.any():
            gathered[batch_idx, valid] = logits[batch_idx, positions[valid]]
    return gathered
