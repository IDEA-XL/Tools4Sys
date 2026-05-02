from __future__ import annotations

import math
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


class HistoryBasedDiversityMemory:
    def __init__(self, config, *, featurize, similarity):
        if not isinstance(config, HBDConfig):
            raise TypeError(f'config must be an HBDConfig, got {type(config)!r}')
        if not callable(featurize):
            raise TypeError('featurize must be callable')
        if not callable(similarity):
            raise TypeError('similarity must be callable')
        self.config = config
        self._featurize = featurize
        self._similarity = similarity
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
        if not self.config.enabled:
            raise RuntimeError('HBD memory apply() called while HBD is disabled')
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
        new_bucket_records: list[tuple[str, object]] = []
        considered_count = 0
        penalized_count = 0
        accepted_existing_count = 0
        created_bucket_count = 0
        bucket_count_before = len(self._buckets)

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
            matched_bucket_idx, best_similarity = self._best_bucket(feature)
            if matched_bucket_idx is None or best_similarity < self.config.similarity_cutoff:
                new_bucket_records.append((index_value, feature))
                created_bucket_count += 1
                continue

            matched_bucket = self._buckets[matched_bucket_idx]
            if matched_bucket.count < self.config.bucket_size:
                bucket_increments[matched_bucket_idx] = bucket_increments.get(matched_bucket_idx, 0) + 1
                accepted_existing_count += 1
                continue

            penalized_count += 1
            final_rewards[idx] = 0.0

        for bucket_idx, increment in bucket_increments.items():
            bucket = self._buckets[bucket_idx]
            bucket.count = min(self.config.bucket_size, bucket.count + increment)
        for index_value, feature in new_bucket_records:
            self._buckets.append(
                _Bucket(
                    index_value=index_value,
                    index_feature=feature,
                    count=1,
                )
            )

        return final_rewards, {
            'enabled': 1.0,
            'eligible_count': float(considered_count),
            'penalized_count': float(penalized_count),
            'accepted_existing_count': float(accepted_existing_count),
            'created_bucket_count': float(created_bucket_count),
            'bucket_count_before': float(bucket_count_before),
            'bucket_count_after': float(len(self._buckets)),
        }

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
    )


def build_sequence_hbd_memory(config):
    return HistoryBasedDiversityMemory(
        config,
        featurize=_normalize_sequence_feature,
        similarity=_normalized_edit_similarity,
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
    return _get_morgan_generator().GetFingerprint(mol)


def _tanimoto_similarity(left_fingerprint, right_fingerprint):
    from rdkit import DataStructs

    return float(DataStructs.TanimotoSimilarity(left_fingerprint, right_fingerprint))


def _normalize_sequence_feature(sequence):
    normalized = str(sequence).strip().upper()
    if not normalized:
        raise ValueError('HBD protein sequence must be non-empty')
    return normalized


@lru_cache(maxsize=4096)
def _normalized_edit_similarity(left_sequence, right_sequence):
    left_sequence = _normalize_sequence_feature(left_sequence)
    right_sequence = _normalize_sequence_feature(right_sequence)
    len_left = len(left_sequence)
    len_right = len(right_sequence)
    dp = [[0] * (len_right + 1) for _ in range(len_left + 1)]
    for left_idx in range(len_left + 1):
        dp[left_idx][0] = left_idx
    for right_idx in range(len_right + 1):
        dp[0][right_idx] = right_idx
    for left_idx in range(1, len_left + 1):
        for right_idx in range(1, len_right + 1):
            substitution_cost = 0 if left_sequence[left_idx - 1] == right_sequence[right_idx - 1] else 1
            dp[left_idx][right_idx] = min(
                dp[left_idx - 1][right_idx] + 1,
                dp[left_idx][right_idx - 1] + 1,
                dp[left_idx - 1][right_idx - 1] + substitution_cost,
            )
    return 1.0 - (dp[len_left][len_right] / float(max(len_left, len_right)))
