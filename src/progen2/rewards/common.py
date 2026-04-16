from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

PROTEIN_ALPHABET = frozenset({'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'})


def validate_batch_size(batch_size, *, field_name='batch_size'):
    try:
        value = int(batch_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field_name} must be a positive integer, got {batch_size!r}') from exc
    if value <= 0:
        raise ValueError(f'{field_name} must be a positive integer, got {batch_size!r}')
    return value


def iter_chunks(items: Sequence[str], batch_size: int) -> Iterable[list[str]]:
    batch_size = validate_batch_size(batch_size)
    for start in range(0, len(items), batch_size):
        yield list(items[start:start + batch_size])


def normalize_protein_sequence(sequence):
    return str(sequence).strip().upper()


def validate_protein_sequence(sequence):
    normalized = normalize_protein_sequence(sequence)
    if not normalized:
        raise ValueError('protein sequence must be non-empty')
    invalid = sorted(set(normalized) - PROTEIN_ALPHABET)
    if invalid:
        raise ValueError(f'protein sequence contains unsupported residues: {invalid}')
    return normalized


def is_valid_protein_sequence(sequence):
    try:
        validate_protein_sequence(sequence)
    except ValueError:
        return False
    return True


def release_model(model, device):
    if model is None:
        return
    device = torch.device(device)
    if device.type != 'cuda':
        return
    model.to('cpu')
    torch.cuda.empty_cache()
