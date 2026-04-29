import pytest
import yaml

from progen2.rl.trainer import (
    _distributed_calibration_batch_sizes,
    _shard_list_for_rank,
    default_reward_batch_size,
    load_config,
)
from progen2.rewards.composite import CompositeProteinReward, build_protein_reward_calibration


class _RecordingScorer:
    def __init__(self, *args, **kwargs):
        self.kwargs = dict(kwargs)


class _ConstantScorer(_RecordingScorer):
    def __init__(self, value, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = float(value)
        self.device = 'cpu'
        self.last_move_to_device_sec = 0.0
        self.last_release_to_cpu_sec = 0.0
        self.calls = 0

    def score_raw(self, sequences):
        self.calls += 1
        return [self.value] * len(sequences)

    def release(self):
        return


def test_load_config_accepts_sgrpo_and_computes_default_reward_batch_size(tmp_path):
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(
        yaml.safe_dump(
            {
                'model_variant': 'progen2_sgrpo',
                'official_code_dir': '/tmp/official',
                'tokenizer_path': '/tmp/tokenizer.json',
                'init_checkpoint_dir': '/tmp/checkpoint',
                'prompt_path': '/tmp/prompts.txt',
                'per_device_prompt_batch_size': 24,
                'num_generations': 4,
                'supergroup_num_groups': 2,
                'rl_algorithm': 'sgrpo',
                'temperature': [0.5, 3.0],
                'reward_compute_every_n_steps': [1, 2, 3, 4],
                'rewards': {
                    'naturalness': {'model_name': 'esm2_t33_650M_UR50D'},
                    'foldability': {},
                    'stability': {'model_name_or_path': '/tmp/temberture', 'base_model_name_or_path': '/tmp/protbert'},
                    'developability': {'model_name_or_path': '/tmp/proteinsol'},
                },
            },
            sort_keys=False,
        )
    )
    config = load_config(config_path)
    assert config.rl_algorithm == 'sgrpo'
    assert config.temperature == [0.5, 3.0]
    assert default_reward_batch_size(config) == 192
    assert config.reward_compute_every_n_steps == {
        'naturalness': 1,
        'foldability': 2,
        'stability': 3,
        'developability': 4,
    }


def test_load_config_rejects_nonpositive_reward_calibration_size(tmp_path):
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(
        yaml.safe_dump(
            {
                'model_variant': 'progen2_sgrpo',
                'official_code_dir': '/tmp/official',
                'tokenizer_path': '/tmp/tokenizer.json',
                'init_checkpoint_dir': '/tmp/checkpoint',
                'prompt_path': '/tmp/prompts.txt',
                'per_device_prompt_batch_size': 2,
                'num_generations': 4,
                'supergroup_num_groups': 2,
                'rl_algorithm': 'sgrpo',
                'reward_calibration_size': 0,
                'rewards': {
                    'naturalness': {'model_name': 'esm2_t33_650M_UR50D'},
                    'foldability': {},
                    'stability': {'model_name_or_path': '/tmp/temberture', 'base_model_name_or_path': '/tmp/protbert'},
                    'developability': {'model_name_or_path': '/tmp/proteinsol'},
                },
            },
            sort_keys=False,
        )
    )
    with pytest.raises(ValueError, match='reward_calibration_size must be positive'):
        load_config(config_path)


def test_distributed_calibration_batch_sizes_are_per_device_and_trim_tail():
    assert _distributed_calibration_batch_sizes(10, 8, 2) == [8, 2]
    assert _distributed_calibration_batch_sizes(32, 8, 4) == [8, 8, 8, 8]
    assert _distributed_calibration_batch_sizes(3, 8, 4) == [3, 0, 0, 0]


def test_shard_list_for_rank_balances_contiguous_ranges():
    items = list(range(10))
    assert _shard_list_for_rank(items, 3, 0) == [0, 1, 2, 3]
    assert _shard_list_for_rank(items, 3, 1) == [4, 5, 6]
    assert _shard_list_for_rank(items, 3, 2) == [7, 8, 9]


def test_build_protein_reward_calibration_uses_only_supported_rewards():
    calibration = build_protein_reward_calibration(
        {
            'naturalness': [0.0, 0.5, 1.0],
            'stability': [1.0, 2.0, 3.0],
            'foldability': [0.2, 0.4],
        }
    )
    assert set(calibration) == {
        'naturalness_q10',
        'naturalness_q90',
        'stability_q10',
        'stability_q90',
    }


def test_composite_reward_uses_shared_default_batch_size(monkeypatch):
    recorded = {}

    def _factory(name):
        def _build(*args, **kwargs):
            scorer = _RecordingScorer(*args, **kwargs)
            recorded[name] = scorer
            return scorer
        return _build

    monkeypatch.setattr('progen2.rewards.composite.ESM2NaturalnessScorer', _factory('naturalness'))
    monkeypatch.setattr('progen2.rewards.composite.ESMFoldFoldabilityScorer', _factory('foldability'))
    monkeypatch.setattr('progen2.rewards.composite.TemBERTureTmScorer', _factory('stability'))
    monkeypatch.setattr('progen2.rewards.composite.ProteinSolScorer', _factory('developability'))

    CompositeProteinReward(
        {
            'naturalness': {'model_name': 'esm2_t33_650M_UR50D'},
            'foldability': {},
            'stability': {'model_name_or_path': '/tmp/temberture', 'base_model_name_or_path': '/tmp/protbert'},
            'developability': {'model_name_or_path': '/tmp/proteinsol'},
        },
        default_reward_batch_size=192,
    )

    assert recorded['naturalness'].kwargs['batch_size'] == 192
    assert recorded['foldability'].kwargs['batch_size'] == 192
    assert recorded['stability'].kwargs['batch_size'] == 192
    assert recorded['developability'].kwargs['batch_size'] == 192


def test_composite_reward_skips_rewards_on_non_scheduled_steps(monkeypatch):
    recorded = {}

    def _factory(name, value):
        def _build(*args, **kwargs):
            scorer = _ConstantScorer(value, *args, **kwargs)
            recorded[name] = scorer
            return scorer
        return _build

    monkeypatch.setattr('progen2.rewards.composite.ESM2NaturalnessScorer', _factory('naturalness', 0.6))
    monkeypatch.setattr('progen2.rewards.composite.ESMFoldFoldabilityScorer', _factory('foldability', 0.7))
    monkeypatch.setattr('progen2.rewards.composite.TemBERTureTmScorer', _factory('stability', 0.8))
    monkeypatch.setattr('progen2.rewards.composite.ProteinSolScorer', _factory('developability', 0.9))
    monkeypatch.setattr(
        'progen2.rewards.composite.score_developability_components',
        lambda raw_values, sequences: {
            'solubility': [0.3] * len(sequences),
            'liability_reward': [0.4] * len(sequences),
            'developability': [0.9] * len(sequences),
        },
    )

    reward = CompositeProteinReward(
        {
            'naturalness': {'model_name': 'esm2_t33_650M_UR50D'},
            'foldability': {},
            'stability': {'model_name_or_path': '/tmp/temberture', 'base_model_name_or_path': '/tmp/protbert'},
            'developability': {'model_name_or_path': '/tmp/proteinsol'},
        },
        default_reward_batch_size=8,
        reward_compute_every_n_steps={
            'naturalness': 2,
            'foldability': 3,
            'stability': 1,
            'developability': 4,
        },
    )
    reward.calibration = {
        'naturalness_q10': 0.0,
        'naturalness_q90': 1.0,
        'stability_q10': 0.0,
        'stability_q90': 1.0,
    }

    details, metrics = reward.score_details(['ACDE', 'FGHI'], step_number=2)

    assert details['naturalness'] == [0.0, 0.0]
    assert details['foldability'] == [0.0, 0.0]
    assert all(abs(value - 0.8) < 1e-6 for value in details['stability'])
    assert details['developability'] == [0.0, 0.0]
    assert all(abs(value - 0.16) < 1e-6 for value in details['total'])
    assert metrics['reward_nat_skipped'] == 1.0
    assert metrics['reward_fold_skipped'] == 1.0
    assert metrics['reward_stab_skipped'] == 0.0
    assert metrics['reward_dev_skipped'] == 1.0
    assert metrics['reward_score_sec_total'] >= 0.0
    assert recorded['naturalness'].calls == 0
    assert recorded['foldability'].calls == 0
    assert recorded['stability'].calls == 1
    assert recorded['developability'].calls == 0
