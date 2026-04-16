import math
from collections import Counter

from progen2.rewards.common import validate_protein_sequence


HYDROPHOBIC_RESIDUES = frozenset({'A', 'V', 'I', 'L', 'M', 'F', 'W', 'Y'})
KYTE_DOOLITTLE = {
    'A': 1.8,
    'C': 2.5,
    'D': -3.5,
    'E': -3.5,
    'F': 2.8,
    'G': -0.4,
    'H': -3.2,
    'I': 4.5,
    'K': -3.9,
    'L': 3.8,
    'M': 1.9,
    'N': -3.5,
    'P': -1.6,
    'Q': -3.5,
    'R': -4.5,
    'S': -0.8,
    'T': -0.7,
    'V': 4.2,
    'W': -0.9,
    'Y': -1.3,
}


def _clip01(value):
    return max(0.0, min(1.0, float(value)))
def tm_like_indicator(sequence):
    sequence = validate_protein_sequence(sequence)
    if len(sequence) < 19:
        return 0.0
    scores = []
    for start in range(0, len(sequence) - 19 + 1):
        window = sequence[start:start + 19]
        scores.append(sum(KYTE_DOOLITTLE[residue] for residue in window) / 19.0)
    max_hydro_19mer = max(scores)
    return _clip01((max_hydro_19mer - 1.6) / (2.6 - 1.6))


def low_complexity_indicator(sequence):
    sequence = validate_protein_sequence(sequence)
    counts = Counter(sequence)
    total = float(len(sequence))
    entropy = 0.0
    for count in counts.values():
        prob = count / total
        entropy -= prob * math.log(prob)
    normalized_entropy = entropy / math.log(20.0)
    return _clip01((0.75 - normalized_entropy) / (0.75 - 0.50))


def hydrophobic_run_indicator(sequence):
    sequence = validate_protein_sequence(sequence)
    longest = 0
    current = 0
    for residue in sequence:
        if residue in HYDROPHOBIC_RESIDUES:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return _clip01((float(longest) - 6.0) / (12.0 - 6.0))


def cys_outlier_indicator(sequence):
    sequence = validate_protein_sequence(sequence)
    cys_fraction = sequence.count('C') / float(len(sequence))
    return _clip01((cys_fraction - 0.05) / (0.12 - 0.05))


def liability_penalty(sequence):
    return (
        0.35 * tm_like_indicator(sequence)
        + 0.25 * low_complexity_indicator(sequence)
        + 0.20 * hydrophobic_run_indicator(sequence)
        + 0.20 * cys_outlier_indicator(sequence)
    )


def liability_reward(sequence):
    return 1.0 - liability_penalty(sequence)
