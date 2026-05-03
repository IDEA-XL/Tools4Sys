from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class HBDConfig:
    enabled: bool = False
    bucket_size: int = 25
    score_threshold_for_memory: float = 0.6
    similarity_cutoff: float = 0.6


def validate_hbd_config(
    *,
    enabled,
    bucket_size,
    score_threshold_for_memory,
    similarity_cutoff,
):
    normalized = HBDConfig(
        enabled=bool(enabled),
        bucket_size=int(bucket_size),
        score_threshold_for_memory=float(score_threshold_for_memory),
        similarity_cutoff=float(similarity_cutoff),
    )
    if normalized.bucket_size <= 0:
        raise ValueError(f'hbd_bucket_size must be positive, got {normalized.bucket_size}')
    if not 0.0 <= normalized.score_threshold_for_memory <= 1.0:
        raise ValueError(
            'hbd_score_threshold_for_memory must be in [0, 1], '
            f'got {normalized.score_threshold_for_memory}'
        )
    if not 0.0 <= normalized.similarity_cutoff <= 1.0:
        raise ValueError(f'hbd_similarity_cutoff must be in [0, 1], got {normalized.similarity_cutoff}')
    return normalized


@dataclass
class _Bucket:
    index_value: str
    index_feature: object
    count: int


@dataclass(frozen=True)
class _MoleculeFeature:
    fingerprint: object
    bit_count: int


@dataclass(frozen=True)
class _PlannedHBDUpdate:
    bucket_count_before: int
    final_rewards: tuple[float, ...]
    bucket_increments: tuple[tuple[int, int], ...]
    new_bucket_values: tuple[str, ...]
    metrics: dict[str, float]

    def to_payload(self):
        return {
            'bucket_count_before': int(self.bucket_count_before),
            'final_rewards': list(self.final_rewards),
            'bucket_increments': [[int(bucket_idx), int(increment)] for bucket_idx, increment in self.bucket_increments],
            'new_bucket_values': list(self.new_bucket_values),
            'metrics': {str(key): float(value) for key, value in self.metrics.items()},
        }


class HistoryBasedDiversityMemory:
    def __init__(self, config, *, featurize, similarity, batch_matcher=None):
        if not isinstance(config, HBDConfig):
            raise TypeError(f'config must be an HBDConfig, got {type(config)!r}')
        if not callable(featurize):
            raise TypeError('featurize must be callable')
        if not callable(similarity):
            raise TypeError('similarity must be callable')
        if batch_matcher is not None and not callable(batch_matcher):
            raise TypeError('batch_matcher must be callable when provided')
        self.config = config
        self._featurize = featurize
        self._similarity = similarity
        self._batch_matcher = batch_matcher
        self._buckets: list[_Bucket] = []

    def reset(self):
        self._buckets = []

    def state_dict(self):
        return {
            'enabled': bool(self.config.enabled),
            'bucket_size': int(self.config.bucket_size),
            'score_threshold_for_memory': float(self.config.score_threshold_for_memory),
            'similarity_cutoff': float(self.config.similarity_cutoff),
            'buckets': [
                {
                    'index_value': bucket.index_value,
                    'count': int(bucket.count),
                }
                for bucket in self._buckets
            ],
        }

    def load_state_dict(self, state):
        if state is None:
            self.reset()
            return
        if not isinstance(state, dict):
            raise TypeError(f'HBD state must be a dict, got {type(state)!r}')

        saved_config = validate_hbd_config(
            enabled=state.get('enabled', False),
            bucket_size=state.get('bucket_size', self.config.bucket_size),
            score_threshold_for_memory=state.get(
                'score_threshold_for_memory',
                self.config.score_threshold_for_memory,
            ),
            similarity_cutoff=state.get('similarity_cutoff', self.config.similarity_cutoff),
        )
        if saved_config != self.config:
            raise ValueError(
                'HBD checkpoint config does not match current config: '
                f'checkpoint={saved_config} current={self.config}'
            )

        buckets = state.get('buckets', [])
        if not isinstance(buckets, list):
            raise TypeError(f'HBD buckets must be a list, got {type(buckets)!r}')
        restored = []
        for bucket in buckets:
            if not isinstance(bucket, dict):
                raise TypeError(f'HBD bucket must be a dict, got {type(bucket)!r}')
            index_value = str(bucket['index_value'])
            count = int(bucket['count'])
            if count <= 0:
                raise ValueError(f'HBD bucket count must be positive, got {count}')
            restored.append(
                _Bucket(
                    index_value=index_value,
                    index_feature=self._featurize(index_value),
                    count=min(count, self.config.bucket_size),
                )
            )
        self._buckets = restored

    def apply(self, items, *, memory_scores, reward_values=None):
        planned_update = self.plan_update(
            items,
            memory_scores=memory_scores,
            reward_values=reward_values,
        )
        self.apply_update(planned_update)
        return list(planned_update.final_rewards), dict(planned_update.metrics)

    def plan_update(self, items, *, memory_scores, reward_values=None):
        if not self.config.enabled:
            raise RuntimeError('HBD memory plan_update() called while HBD is disabled')
        if reward_values is None:
            reward_values = memory_scores
        if len(items) != len(memory_scores):
            raise ValueError(
                f'HBD items/memory_scores length mismatch: {len(items)} vs {len(memory_scores)}'
            )
        if len(items) != len(reward_values):
            raise ValueError(
                f'HBD items/reward_values length mismatch: {len(items)} vs {len(reward_values)}'
            )

        final_rewards = [float(reward) for reward in reward_values]
        bucket_increments: dict[int, int] = {}
        new_bucket_values: list[str] = []
        considered_count = 0
        penalized_count = 0
        accepted_existing_count = 0
        created_bucket_count = 0
        bucket_count_before = len(self._buckets)
        eligible_records: list[tuple[int, str, object]] = []

        for idx, (item, score, reward) in enumerate(zip(items, memory_scores, reward_values)):
            score_value = float(score)
            reward_value = float(reward)
            if math.isnan(score_value):
                raise ValueError('HBD memory score must not be NaN')
            if math.isinf(score_value) and score_value > 0.0:
                raise ValueError(f'HBD memory score must not be +inf, got {score_value}')
            if not math.isfinite(reward_value):
                raise ValueError(f'HBD reward value must be finite, got {reward_value}')
            if score_value < self.config.score_threshold_for_memory:
                continue

            considered_count += 1
            index_value = str(item)
            feature = self._featurize(index_value)
            eligible_records.append((idx, index_value, feature))

        match_results = self._resolve_match_results(eligible_records)
        for (idx, index_value, _feature), (matched_bucket_idx, best_similarity) in zip(
            eligible_records,
            match_results,
        ):
            if matched_bucket_idx is None or best_similarity < self.config.similarity_cutoff:
                new_bucket_values.append(index_value)
                created_bucket_count += 1
                continue

            matched_bucket = self._buckets[matched_bucket_idx]
            if matched_bucket.count < self.config.bucket_size:
                bucket_increments[matched_bucket_idx] = bucket_increments.get(matched_bucket_idx, 0) + 1
                accepted_existing_count += 1
                continue

            penalized_count += 1
            final_rewards[idx] = 0.0

        bucket_count_after = bucket_count_before + len(new_bucket_values)
        return _PlannedHBDUpdate(
            bucket_count_before=int(bucket_count_before),
            final_rewards=tuple(final_rewards),
            bucket_increments=tuple(sorted((int(bucket_idx), int(increment)) for bucket_idx, increment in bucket_increments.items())),
            new_bucket_values=tuple(new_bucket_values),
            metrics={
                'enabled': 1.0,
                'eligible_count': float(considered_count),
                'penalized_count': float(penalized_count),
                'accepted_existing_count': float(accepted_existing_count),
                'created_bucket_count': float(created_bucket_count),
                'bucket_count_before': float(bucket_count_before),
                'bucket_count_after': float(bucket_count_after),
            },
        )

    def apply_update(self, planned_update):
        normalized = self._normalize_planned_update(planned_update)
        if normalized.bucket_count_before != len(self._buckets):
            raise ValueError(
                'HBD bucket count mismatch before apply_update: '
                f'expected {normalized.bucket_count_before}, got {len(self._buckets)}'
            )
        for bucket_idx, increment in normalized.bucket_increments:
            if not 0 <= bucket_idx < len(self._buckets):
                raise IndexError(
                    f'HBD bucket increment index out of range: {bucket_idx} for {len(self._buckets)} buckets'
                )
            if increment <= 0:
                raise ValueError(f'HBD bucket increment must be positive, got {increment}')
            bucket = self._buckets[bucket_idx]
            bucket.count = min(self.config.bucket_size, bucket.count + increment)
        for index_value in normalized.new_bucket_values:
            self._buckets.append(
                _Bucket(
                    index_value=index_value,
                    index_feature=self._featurize(index_value),
                    count=1,
                )
            )

    def _normalize_planned_update(self, planned_update):
        if isinstance(planned_update, _PlannedHBDUpdate):
            return planned_update
        if not isinstance(planned_update, dict):
            raise TypeError(f'HBD planned update must be a dict or _PlannedHBDUpdate, got {type(planned_update)!r}')
        final_rewards = planned_update.get('final_rewards')
        bucket_increments = planned_update.get('bucket_increments')
        new_bucket_values = planned_update.get('new_bucket_values')
        metrics = planned_update.get('metrics')
        if not isinstance(final_rewards, list):
            raise TypeError(f'HBD final_rewards must be a list, got {type(final_rewards)!r}')
        if not isinstance(bucket_increments, list):
            raise TypeError(f'HBD bucket_increments must be a list, got {type(bucket_increments)!r}')
        if not isinstance(new_bucket_values, list):
            raise TypeError(f'HBD new_bucket_values must be a list, got {type(new_bucket_values)!r}')
        if not isinstance(metrics, dict):
            raise TypeError(f'HBD metrics must be a dict, got {type(metrics)!r}')
        return _PlannedHBDUpdate(
            bucket_count_before=int(planned_update['bucket_count_before']),
            final_rewards=tuple(float(value) for value in final_rewards),
            bucket_increments=tuple((int(bucket_idx), int(increment)) for bucket_idx, increment in bucket_increments),
            new_bucket_values=tuple(str(value) for value in new_bucket_values),
            metrics={str(key): float(value) for key, value in metrics.items()},
        )

    def _resolve_match_results(self, eligible_records):
        if not eligible_records:
            return []
        features = [feature for _, _, feature in eligible_records]
        if self._batch_matcher is not None and self._buckets:
            return self._batch_matcher(
                features,
                tuple(bucket.index_feature for bucket in self._buckets),
                self.config.similarity_cutoff,
            )
        return [self._best_bucket(feature) for feature in features]

    def _best_bucket(self, feature):
        best_bucket_idx = None
        best_similarity = float('-inf')
        for bucket_idx, bucket in enumerate(self._buckets):
            similarity = float(self._similarity(feature, bucket.index_feature))
            if not 0.0 <= similarity <= 1.0:
                raise ValueError(f'HBD similarity must be in [0, 1], got {similarity}')
            if similarity > best_similarity:
                best_bucket_idx = bucket_idx
                best_similarity = similarity
        return best_bucket_idx, best_similarity


def build_molecule_hbd_memory(config):
    return HistoryBasedDiversityMemory(
        config,
        featurize=_morgan_fingerprint_from_smiles,
        similarity=_tanimoto_similarity,
        batch_matcher=_build_molecule_batch_matcher(),
    )


def build_sequence_hbd_memory(config):
    return HistoryBasedDiversityMemory(
        config,
        featurize=_normalize_sequence_feature,
        similarity=_normalized_edit_similarity,
        batch_matcher=_build_sequence_batch_matcher(),
    )


@lru_cache(maxsize=1)
def _get_morgan_generator():
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


@lru_cache(maxsize=65536)
def _morgan_fingerprint_from_smiles(smiles):
    from rdkit.Chem import MolFromSmiles

    normalized = str(smiles).strip()
    if not normalized:
        raise ValueError('HBD SMILES must be non-empty')
    mol = MolFromSmiles(normalized, sanitize=True)
    if mol is None:
        raise ValueError(f'Failed to parse HBD SMILES: {normalized!r}')
    fingerprint = _get_morgan_generator().GetFingerprint(mol)
    return _MoleculeFeature(
        fingerprint=fingerprint,
        bit_count=int(fingerprint.GetNumOnBits()),
    )


def _tanimoto_similarity(left_fingerprint, right_fingerprint):
    from rdkit import DataStructs

    return float(DataStructs.TanimotoSimilarity(left_fingerprint.fingerprint, right_fingerprint.fingerprint))


def _build_molecule_batch_matcher():
    worker_count = _get_hbd_cpu_worker_count()

    def _match(features, bucket_features, similarity_cutoff):
        return _find_best_molecule_bucket_matches(
            features,
            bucket_features,
            similarity_cutoff=similarity_cutoff,
            worker_count=worker_count,
        )

    return _match


def _normalize_sequence_feature(sequence):
    normalized = str(sequence).strip().upper()
    if not normalized:
        raise ValueError('HBD protein sequence must be non-empty')
    return normalized


def _build_sequence_batch_matcher():
    _, _, _ = _get_sequence_distance_backend()
    worker_count = _get_hbd_cpu_worker_count()

    def _match(features, bucket_features, similarity_cutoff):
        return _find_best_sequence_bucket_matches(
            features,
            bucket_features,
            similarity_cutoff=similarity_cutoff,
            worker_count=worker_count,
        )

    return _match


@lru_cache(maxsize=1)
def _get_numpy_backend():
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            'Molecule HBD requires numpy for bulk similarity matching. '
            'Install the project environment from genmol/env/requirements*.'
        ) from exc
    return np


@lru_cache(maxsize=1)
def _get_sequence_distance_backend():
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            'Sequence HBD requires numpy for the rapidfuzz distance backend. '
            'Install the project environment from genmol/env/requirements*.'
        ) from exc
    try:
        from rapidfuzz import process as rapidfuzz_process
        from rapidfuzz.distance import Levenshtein
    except ImportError as exc:
        raise ImportError(
            'Sequence HBD requires rapidfuzz. Add it to the active environment before enabling HBD for progen2.'
        ) from exc
    return np, rapidfuzz_process, Levenshtein


@lru_cache(maxsize=1)
def _get_hbd_cpu_worker_count():
    affinity_count = None
    if hasattr(os, 'sched_getaffinity'):
        try:
            affinity_count = len(os.sched_getaffinity(0))
        except OSError:
            affinity_count = None
    if affinity_count is not None and affinity_count > 0:
        return affinity_count
    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count <= 0:
        raise RuntimeError('failed to determine available CPU worker count for sequence HBD')
    return int(cpu_count)


def _find_best_molecule_bucket_matches(features, bucket_features, *, similarity_cutoff, worker_count):
    if not features:
        return []
    if not bucket_features:
        return [(None, float('-inf')) for _ in features]
    if worker_count <= 0:
        raise ValueError(f'molecule HBD worker_count must be positive, got {worker_count}')

    bucket_bit_counts = [int(feature.bit_count) for feature in bucket_features]
    grouped_queries: dict[int, list[tuple[int, _MoleculeFeature]]] = {}
    for feature_idx, feature in enumerate(features):
        grouped_queries.setdefault(int(feature.bit_count), []).append((feature_idx, feature))

    candidate_cache: dict[int, tuple[tuple[int, ...], tuple[object, ...]]] = {}
    work_items: list[tuple[list[tuple[int, _MoleculeFeature]], tuple[int, ...], tuple[object, ...]]] = []
    for query_bit_count, query_group in grouped_queries.items():
        candidate_indices = tuple(
            bucket_idx
            for bucket_idx, bucket_bit_count in enumerate(bucket_bit_counts)
            if _molecule_bitcount_upper_bound(query_bit_count, bucket_bit_count) >= similarity_cutoff
        )
        if not candidate_indices:
            continue
        candidate_fingerprints = candidate_cache.get(query_bit_count)
        if candidate_fingerprints is None:
            candidate_fingerprints = (
                candidate_indices,
                tuple(bucket_features[bucket_idx].fingerprint for bucket_idx in candidate_indices),
            )
            candidate_cache[query_bit_count] = candidate_fingerprints
        max_group_workers = min(worker_count, len(query_group))
        chunk_size = max(1, math.ceil(len(query_group) / max_group_workers))
        for start in range(0, len(query_group), chunk_size):
            work_items.append(
                (
                    query_group[start:start + chunk_size],
                    candidate_fingerprints[0],
                    candidate_fingerprints[1],
                )
            )

    match_results: list[tuple[int | None, float]] = [(None, float('-inf')) for _ in features]
    if not work_items:
        return match_results
    if worker_count == 1 or len(work_items) == 1:
        for query_chunk, candidate_indices, candidate_fingerprints in work_items:
            for feature_idx, match_result in _match_molecule_query_chunk(
                query_chunk,
                candidate_indices,
                candidate_fingerprints,
                similarity_cutoff=similarity_cutoff,
            ):
                match_results[feature_idx] = match_result
        return match_results

    with ThreadPoolExecutor(
        max_workers=min(worker_count, len(work_items)),
        thread_name_prefix='hbd_molecule_match',
    ) as executor:
        futures = [
            executor.submit(
                _match_molecule_query_chunk,
                query_chunk,
                candidate_indices,
                candidate_fingerprints,
                similarity_cutoff=similarity_cutoff,
            )
            for query_chunk, candidate_indices, candidate_fingerprints in work_items
        ]
        for future in futures:
            for feature_idx, match_result in future.result():
                match_results[feature_idx] = match_result
    return match_results


def _match_molecule_query_chunk(query_chunk, candidate_indices, candidate_fingerprints, *, similarity_cutoff):
    from rdkit import DataStructs

    np = _get_numpy_backend()
    results = []
    for feature_idx, feature in query_chunk:
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(feature.fingerprint, candidate_fingerprints),
            dtype=np.float32,
        )
        if similarities.size == 0:
            results.append((feature_idx, (None, float('-inf'))))
            continue
        best_column = int(similarities.argmax())
        best_similarity = float(similarities[best_column])
        if not 0.0 <= best_similarity <= 1.0:
            raise ValueError(f'HBD similarity must be in [0, 1], got {best_similarity}')
        if best_similarity >= similarity_cutoff:
            results.append((feature_idx, (candidate_indices[best_column], best_similarity)))
            continue
        results.append((feature_idx, (None, float('-inf'))))
    return results


def _molecule_bitcount_upper_bound(left_bit_count, right_bit_count):
    left = int(left_bit_count)
    right = int(right_bit_count)
    if left < 0 or right < 0:
        raise ValueError(
            f'Molecule HBD bit counts must be non-negative, got left={left} right={right}'
        )
    denominator = max(left, right)
    if denominator == 0:
        return 1.0
    return float(min(left, right)) / float(denominator)


def _find_best_sequence_bucket_matches(features, bucket_features, *, similarity_cutoff, worker_count):
    if not features:
        return []
    if not bucket_features:
        return [(None, float('-inf')) for _ in features]
    np, rapidfuzz_process, Levenshtein = _get_sequence_distance_backend()
    bucket_lengths = [len(feature) for feature in bucket_features]
    grouped_queries: dict[int, list[tuple[int, str]]] = {}
    for feature_idx, feature in enumerate(features):
        grouped_queries.setdefault(len(feature), []).append((feature_idx, feature))

    match_results: list[tuple[int | None, float]] = [(None, float('-inf')) for _ in features]
    for query_length, query_group in grouped_queries.items():
        candidate_indices = [
            bucket_idx
            for bucket_idx, bucket_length in enumerate(bucket_lengths)
            if _length_gap_can_reach_similarity_cutoff(query_length, bucket_length, similarity_cutoff)
        ]
        if not candidate_indices:
            continue
        candidate_sequences = [bucket_features[bucket_idx] for bucket_idx in candidate_indices]
        similarities = rapidfuzz_process.cdist(
            [feature for _, feature in query_group],
            candidate_sequences,
            scorer=Levenshtein.normalized_similarity,
            score_cutoff=similarity_cutoff,
            workers=worker_count,
            dtype=np.float32,
        )
        for row_idx, (feature_idx, _feature) in enumerate(query_group):
            row = similarities[row_idx]
            if row.size == 0:
                continue
            best_column = int(row.argmax())
            best_similarity = float(row[best_column])
            if best_similarity >= similarity_cutoff:
                match_results[feature_idx] = (candidate_indices[best_column], best_similarity)
    return match_results


def _length_gap_can_reach_similarity_cutoff(left_length, right_length, similarity_cutoff):
    minimum_distance = abs(int(left_length) - int(right_length))
    max_distance = math.floor((1.0 - float(similarity_cutoff)) * max(int(left_length), int(right_length)) + 1e-12)
    return minimum_distance <= max_distance


@lru_cache(maxsize=4096)
def _normalized_edit_similarity(left_sequence, right_sequence):
    left_sequence = _normalize_sequence_feature(left_sequence)
    right_sequence = _normalize_sequence_feature(right_sequence)
    _, _, Levenshtein = _get_sequence_distance_backend()
    return float(Levenshtein.normalized_similarity(left_sequence, right_sequence))
