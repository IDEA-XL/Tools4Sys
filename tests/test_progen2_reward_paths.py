from pathlib import Path

from progen2.rewards.developability import _parse_scaled_sol_scores, _resolve_proteinsol_root
from progen2.rewards.stability import _resolve_temberture_root


def test_resolve_temberture_root_accepts_repo_root(tmp_path):
    replica = tmp_path / 'temBERTure' / 'temBERTure_TM' / 'replica1'
    replica.mkdir(parents=True)
    assert _resolve_temberture_root(tmp_path) == tmp_path / 'temBERTure'


def test_resolve_proteinsol_root_accepts_parent_dir(tmp_path):
    bundle = tmp_path / 'protein-sol-sequence-prediction-software'
    bundle.mkdir(parents=True)
    (bundle / 'multiple_prediction_wrapper_export.sh').write_text('#!/bin/bash\n')
    assert _resolve_proteinsol_root(tmp_path) == bundle


def test_parse_scaled_sol_scores_reads_scaled_column(tmp_path):
    prediction_path = tmp_path / 'seq_prediction.txt'
    prediction_path.write_text(
        'HEADERS PREDICTIONS LINE,ID,percent-sol,scaled-sol,population-sol,pI\n'
        'SEQUENCE PREDICTIONS,>seq_0,41.535,0.336,0.446,5.520\n'
        'SEQUENCE PREDICTIONS,>seq_1,64.448,0.548,0.446,10.640\n'
    )
    assert _parse_scaled_sol_scores(prediction_path) == {
        'seq_0': 0.336,
        'seq_1': 0.548,
    }
