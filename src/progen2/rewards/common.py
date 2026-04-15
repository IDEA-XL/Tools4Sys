from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch


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


def release_model(model, device):
    if model is None:
        return
    device = torch.device(device)
    if device.type != 'cuda':
        return
    model.to('cpu')
    torch.cuda.empty_cache()
