from progen2.evaluation import classify_protein_sequence, compute_group_diversity_rewards
from progen2.rewards.developability import score_developability_components


def test_classify_protein_sequence_marks_empty_as_invalid():
    result = classify_protein_sequence('')
    assert result == {'sequence': None, 'is_valid': False, 'invalid_reason': 'empty'}


def test_classify_protein_sequence_normalizes_valid_sequences():
    result = classify_protein_sequence(' acde ')
    assert result == {'sequence': 'ACDE', 'is_valid': True, 'invalid_reason': None}


def test_compute_group_diversity_rewards_filters_invalid_sequences_per_group():
    rewards = compute_group_diversity_rewards(
        ['ACDE', 'WXYZ', 'AAAA', 'CCCC'],
        group_size=2,
    )
    assert rewards[0] == 0.0
    assert rewards[1] > 0.0


def test_score_developability_components_returns_weighted_components():
    components = score_developability_components([0.5], ['ACDE'])
    assert components['solubility'] == [0.5]
    assert len(components['liability_reward']) == 1
    assert len(components['developability']) == 1
    assert components['developability'][0] == 0.8 * components['solubility'][0] + 0.2 * components['liability_reward'][0]
