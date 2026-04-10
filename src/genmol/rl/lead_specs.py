import glob
import json
import random
from bisect import bisect_right
from dataclasses import asdict, dataclass
from functools import lru_cache

import pandas as pd
import pyarrow.parquet as pq
import safe as sf


@dataclass(frozen=True)
class LeadOptSpec:
    seed_smiles: str
    mutation_seed: int
    generation_temperature: float
    randomness: float
    min_seed_len: int


@dataclass(frozen=True)
class SeedShardManifest:
    path: str
    num_rows: int
    row_group_offsets: tuple[int, ...]
    source_column: str


def _resolve_seed_paths(seed_data_glob):
    paths = tuple(sorted(glob.glob(seed_data_glob)))
    if not paths:
        raise FileNotFoundError(f'No seed parquet files matched: {seed_data_glob}')
    return paths


@lru_cache(maxsize=8)
def build_seed_manifest(seed_data_glob):
    manifests = []
    for path in _resolve_seed_paths(seed_data_glob):
        parquet = pq.ParquetFile(path)
        column_names = set(parquet.schema_arrow.names)
        if 'smiles' in column_names:
            source_column = 'smiles'
        elif 'input' in column_names:
            source_column = 'input'
        else:
            raise ValueError(f'Expected parquet shard to contain "smiles" or "input" column: {path}')

        offsets = []
        total = 0
        for row_group_idx in range(parquet.metadata.num_row_groups):
            offsets.append(total)
            total += parquet.metadata.row_group(row_group_idx).num_rows
        if total <= 0:
            raise ValueError(f'Parquet shard contains no rows: {path}')

        manifests.append(
            SeedShardManifest(
                path=path,
                num_rows=total,
                row_group_offsets=tuple(offsets),
                source_column=source_column,
            )
        )
    return tuple(manifests)


def validate_seed_data(seed_data_glob):
    manifests = build_seed_manifest(seed_data_glob)
    if not manifests:
        raise ValueError(f'No valid seed parquet shards found: {seed_data_glob}')
    return manifests


@lru_cache(maxsize=8)
def load_seed_smiles(seed_data_glob):
    manifests = build_seed_manifest(seed_data_glob)
    smiles = []
    for manifest in manifests:
        frame = pd.read_parquet(manifest.path, columns=[manifest.source_column])
        if manifest.source_column == 'smiles':
            shard_smiles = frame['smiles'].dropna().astype(str).tolist()
        else:
            shard_smiles = []
            for row_idx, seed_safe in enumerate(frame['input'].dropna().astype(str).tolist()):
                try:
                    decoded = sf.decode(seed_safe)
                except Exception as exc:
                    raise ValueError(
                        f'Failed to decode SAFE seed from {manifest.path} row {row_idx}: {seed_safe!r}'
                    ) from exc
                if not decoded:
                    continue
                shard_smiles.append(decoded)
        smiles.extend(item for item in shard_smiles if item)
    if not smiles:
        raise ValueError(f'No non-empty smiles found in seed parquet shards: {[item.path for item in manifests]}')
    return tuple(smiles)


def _row_group_index(offsets, row_index):
    idx = bisect_right(offsets, row_index) - 1
    if idx < 0:
        raise ValueError(f'Failed to map row index {row_index} into row-group offsets')
    return idx


def _decode_seed_value(source_column, raw_value, path, row_group_idx, row_in_group):
    if raw_value is None:
        raise ValueError(
            f'Encountered empty seed value while sampling shard {path} row_group={row_group_idx} row={row_in_group}'
        )
    if source_column == 'smiles':
        decoded = str(raw_value)
    else:
        try:
            decoded = sf.decode(str(raw_value))
        except Exception as exc:
            raise ValueError(
                f'Failed to decode SAFE seed from {path} row_group={row_group_idx} row={row_in_group}: {raw_value!r}'
            ) from exc
    if not decoded:
        raise ValueError(
            f'Encountered empty decoded seed while sampling shard {path} row_group={row_group_idx} row={row_in_group}'
        )
    return decoded


def sample_seed_smiles(num_samples, seed_data_glob, seed):
    if num_samples <= 0:
        raise ValueError('num_samples must be positive')

    manifests = build_seed_manifest(seed_data_glob)
    total_rows = sum(item.num_rows for item in manifests)
    cumulative_rows = []
    running_total = 0
    for item in manifests:
        running_total += item.num_rows
        cumulative_rows.append(running_total)

    rng = random.Random(seed)
    sampled_positions = []
    for sample_idx in range(num_samples):
        global_row_index = rng.randrange(total_rows)
        shard_idx = bisect_right(cumulative_rows, global_row_index)
        shard_manifest = manifests[shard_idx]
        shard_base = 0 if shard_idx == 0 else cumulative_rows[shard_idx - 1]
        shard_row_index = global_row_index - shard_base
        row_group_idx = _row_group_index(shard_manifest.row_group_offsets, shard_row_index)
        row_group_start = shard_manifest.row_group_offsets[row_group_idx]
        row_in_group = shard_row_index - row_group_start
        sampled_positions.append((sample_idx, shard_manifest, row_group_idx, row_in_group))

    grouped = {}
    for sample_idx, manifest, row_group_idx, row_in_group in sampled_positions:
        grouped.setdefault((manifest.path, row_group_idx), []).append((sample_idx, manifest, row_in_group))

    sampled_smiles = [None] * num_samples
    parquet_cache = {}
    for (path, row_group_idx), positions in grouped.items():
        parquet = parquet_cache.get(path)
        if parquet is None:
            parquet = pq.ParquetFile(path)
            parquet_cache[path] = parquet
        manifest = positions[0][1]
        table = parquet.read_row_group(row_group_idx, columns=[manifest.source_column])
        values = table.column(manifest.source_column).to_pylist()
        for sample_idx, manifest, row_in_group in positions:
            sampled_smiles[sample_idx] = _decode_seed_value(
                manifest.source_column,
                values[row_in_group],
                path,
                row_group_idx,
                row_in_group,
            )

    if any(item is None for item in sampled_smiles):
        raise RuntimeError('Seed sampling produced missing values')
    return tuple(sampled_smiles)


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

    rng = random.Random(seed)
    sampled_seed_smiles = sample_seed_smiles(
        num_samples=num_groups,
        seed_data_glob=seed_data_glob,
        seed=seed,
    )
    specs = []
    for seed_smiles in sampled_seed_smiles:
        specs.append(
            LeadOptSpec(
                seed_smiles=seed_smiles,
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
