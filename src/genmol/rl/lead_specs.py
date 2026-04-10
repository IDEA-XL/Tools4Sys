import glob
import json
import random
from dataclasses import asdict, dataclass
from functools import lru_cache

import pandas as pd
import safe as sf


@dataclass(frozen=True)
class LeadOptSpec:
    seed_smiles: str
    mutation_seed: int
    generation_temperature: float
    randomness: float
    min_seed_len: int


def _resolve_seed_paths(seed_data_glob):
    paths = tuple(sorted(glob.glob(seed_data_glob)))
    if not paths:
        raise FileNotFoundError(f'No seed parquet files matched: {seed_data_glob}')
    return paths


@lru_cache(maxsize=8)
def load_seed_smiles(seed_data_glob):
    seed_paths = _resolve_seed_paths(seed_data_glob)
    smiles = []
    for path in seed_paths:
        frame = pd.read_parquet(path)
        if 'smiles' in frame.columns:
            shard_smiles = frame['smiles'].dropna().astype(str).tolist()
        elif 'input' in frame.columns:
            shard_smiles = []
            for row_idx, seed_safe in enumerate(frame['input'].dropna().astype(str).tolist()):
                try:
                    decoded = sf.decode(seed_safe)
                except Exception as exc:
                    raise ValueError(
                        f'Failed to decode SAFE seed from {path} row {row_idx}: {seed_safe!r}'
                    ) from exc
                if not decoded:
                    continue
                shard_smiles.append(decoded)
        else:
            raise ValueError(f'Expected parquet shard to contain "smiles" or "input" column: {path}')
        smiles.extend(item for item in shard_smiles if item)
    if not smiles:
        raise ValueError(f'No non-empty smiles found in seed parquet shards: {seed_paths}')
    return tuple(smiles)


def sample_group_specs(
    num_groups,
    seed_data_glob,
    generation_temperature,
    randomness,
    min_seed_len,
    seed,
):
    if num_groups <= 0:
        raise ValueError('num_groups must be positive')

    all_seed_smiles = load_seed_smiles(seed_data_glob)
    rng = random.Random(seed)
    specs = []
    for _ in range(num_groups):
        specs.append(
            LeadOptSpec(
                seed_smiles=rng.choice(all_seed_smiles),
                mutation_seed=rng.randrange(2**31),
                generation_temperature=generation_temperature,
                randomness=randomness,
                min_seed_len=min_seed_len,
            )
        )
    return specs


def expand_group_specs(group_specs, num_generations):
    if num_generations <= 1:
        raise ValueError('num_generations must be greater than 1')

    expanded = []
    for spec in group_specs:
        expanded.extend([spec] * num_generations)
    return expanded


def serialize_specs(specs):
    return json.dumps([asdict(spec) for spec in specs], sort_keys=True)


def deserialize_specs(payload):
    return [LeadOptSpec(**item) for item in json.loads(payload)]
