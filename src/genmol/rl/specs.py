import os
import pickle
import random
from dataclasses import dataclass
from functools import lru_cache


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))


@dataclass(frozen=True)
class DeNovoSpec:
    add_seq_len: int
    softmax_temp: float
    randomness: float
    min_add_len: int


@lru_cache(maxsize=1)
def load_length_distribution(path=None):
    length_path = path or os.path.join(ROOT_DIR, 'data', 'len.pk')
    with open(length_path, 'rb') as handle:
        return tuple(pickle.load(handle))


def sample_add_seq_len(min_add_len, rng, length_path=None):
    seq_len_list = load_length_distribution(length_path)
    sampled_total_length = rng.choice(seq_len_list)
    return max(sampled_total_length - 2, min_add_len)


def sample_group_specs(groups_per_rank, softmax_temp, randomness, min_add_len, seed, length_path=None):
    rng = random.Random(seed)
    specs = []
    for _ in range(groups_per_rank):
        specs.append(
            DeNovoSpec(
                add_seq_len=sample_add_seq_len(min_add_len=min_add_len, rng=rng, length_path=length_path),
                softmax_temp=softmax_temp,
                randomness=randomness,
                min_add_len=min_add_len,
            )
        )
    return specs
