import random


def normalize_scalar_or_range(value, *, name, min_exclusive=None):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a number or [low, high], got bool')
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f'{name} range must have exactly two values, got {value!r}')
        low, high = value
        if isinstance(low, bool) or isinstance(high, bool):
            raise ValueError(f'{name} range values must be numeric, got {value!r}')
        try:
            low = float(low)
            high = float(high)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{name} range values must be numeric, got {value!r}') from exc
        if not low < high:
            raise ValueError(f'{name} range must satisfy low < high, got {value!r}')
        if min_exclusive is not None and not low > float(min_exclusive):
            raise ValueError(f'{name} range lower bound must be > {min_exclusive}, got {low}')
        return [low, high]

    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a number or [low, high], got {value!r}') from exc
    if min_exclusive is not None and not scalar > float(min_exclusive):
        raise ValueError(f'{name} must be > {min_exclusive}, got {scalar}')
    return scalar


def sample_scalar_or_range(value, rng: random.Random, *, name, min_exclusive=None):
    normalized = normalize_scalar_or_range(value, name=name, min_exclusive=min_exclusive)
    if isinstance(normalized, list):
        return rng.uniform(normalized[0], normalized[1])
    return normalized
