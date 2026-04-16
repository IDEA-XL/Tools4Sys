from genmol.mm.evaluation import OfficialMoleculeMetricSuite, order_preserving_unique, select_manifest_entries


def test_order_preserving_unique_keeps_first_occurrence():
    assert order_preserving_unique(['A', 'B', 'A', 'C', 'B']) == ['A', 'B', 'C']


def test_select_manifest_entries_returns_all_when_num_pockets_is_none():
    entries = [{'source_index': idx} for idx in range(3)]
    assert select_manifest_entries(entries, None, seed=42) == entries


def test_official_molecule_metric_suite_matches_expected_denovo_logic():
    smiles_list = ['A', None, 'A', 'B']
    qed_map = {'A': 0.7, 'B': 0.5}
    sa_map = {'A': 3.0, 'B': 5.0}

    class FakeOracle:
        def __init__(self, values):
            self.values = values

        def __call__(self, smiles):
            return [self.values[item] for item in smiles]

    def fake_diversity(smiles):
        assert smiles == ['A', 'B']
        return 0.25

    suite = OfficialMoleculeMetricSuite(
        qed_oracle=FakeOracle(qed_map),
        sa_oracle=FakeOracle(sa_map),
        diversity_evaluator=fake_diversity,
    )
    summary = suite.summarize(smiles_list)

    assert summary['num_valid'] == 3
    assert summary['num_unique_valid'] == 2
    assert summary['official_validity'] == 0.75
    assert summary['official_uniqueness'] == 2 / 3
    assert summary['official_quality'] == 0.25
    assert summary['official_diversity'] == 0.25
    assert summary['official_qed_mean'] == 0.6
    assert summary['official_sa_mean'] == 4.0
