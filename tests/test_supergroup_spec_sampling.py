import pickle

import pytest

from genmol.rl.specs import sample_supergroup_shared_specs


def test_sample_supergroup_shared_specs_repeats_spec_within_each_supergroup(tmp_path):
    length_path = tmp_path / 'lengths.pk'
    with length_path.open('wb') as handle:
        pickle.dump([70, 71, 72, 73], handle)

    specs = sample_supergroup_shared_specs(
        num_groups=8,
        supergroup_num_groups=4,
        generation_temperature=1.0,
        randomness=0.3,
        min_add_len=60,
        seed=123,
        length_path=str(length_path),
    )

    assert len(specs) == 8
    for start in range(0, len(specs), 4):
        block = specs[start:start + 4]
        assert all(spec == block[0] for spec in block[1:])


def test_sample_supergroup_shared_specs_requires_exact_divisibility(tmp_path):
    length_path = tmp_path / 'lengths.pk'
    with length_path.open('wb') as handle:
        pickle.dump([70, 71, 72], handle)

    with pytest.raises(ValueError, match='num_groups must be divisible by supergroup_num_groups'):
        sample_supergroup_shared_specs(
            num_groups=10,
            supergroup_num_groups=4,
            generation_temperature=1.0,
            randomness=0.3,
            min_add_len=60,
            seed=123,
            length_path=str(length_path),
        )
