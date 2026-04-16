from genmol.mm.docking import DockingRecord, summarize_docking_records
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


def test_summarize_docking_records_uses_repo_failure_sentinel():
    records = [
        DockingRecord(
            score=-8.5,
            is_success=True,
            error=None,
            receptor_pdb_path='rec_a.pdb',
            receptor_pdbqt_path='rec_a.pdbqt',
            native_ligand_path='lig_a.sdf',
            center_x=1.0,
            center_y=2.0,
            center_z=3.0,
            size_x=18.0,
            size_y=19.0,
            size_z=20.0,
        ),
        DockingRecord(
            score=99.9,
            is_success=False,
            error='dock failed',
            receptor_pdb_path='rec_b.pdb',
            receptor_pdbqt_path='rec_b.pdbqt',
            native_ligand_path='lig_b.sdf',
            center_x=1.0,
            center_y=2.0,
            center_z=3.0,
            size_x=18.0,
            size_y=19.0,
            size_z=20.0,
        ),
    ]
    summary = summarize_docking_records(records)
    assert summary['num_docked'] == 1
    assert summary['docking_success_fraction'] == 0.5
    assert summary['docking_score_mean'] == 45.7
    assert summary['docking_score_median'] == 45.7
