import math

from progen2.rewards.liability import (
    cys_outlier_indicator,
    hydrophobic_run_indicator,
    liability_penalty,
    low_complexity_indicator,
    tm_like_indicator,
)


def test_tm_like_indicator_is_zero_for_short_sequences():
    assert tm_like_indicator('ACDEFGHIK') == 0.0


def test_low_complexity_indicator_detects_repetitive_sequence():
    assert low_complexity_indicator('AAAAAAAAAAAA') > 0.9


def test_hydrophobic_run_indicator_detects_long_runs():
    assert hydrophobic_run_indicator('AAAAAAAVVVVVV') > 0.9


def test_cys_outlier_indicator_detects_high_cysteine_fraction():
    assert cys_outlier_indicator('CCCCCCCCCCAAAA') > 0.9


def test_liability_penalty_stays_in_unit_interval():
    value = liability_penalty('MKTAYIAKQRQISFVKSHFSRQDILDLWQ')
    assert 0.0 <= value <= 1.0
