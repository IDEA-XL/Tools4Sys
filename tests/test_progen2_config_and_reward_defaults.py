import yaml

from progen2.rl.trainer import default_reward_batch_size, load_config
from progen2.rewards.composite import CompositeProteinReward


class _RecordingScorer:
    def __init__(self, *args, **kwargs):
        self.kwargs = dict(kwargs)


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
    assert default_reward_batch_size(config) == 192


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
