import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from genmol.mm.docking import (
    DockingRecord,
    _ligand_center_of_mass,
    _translate_ligand_to_center,
    protein_relative_path_from_ligand_filename,
    summarize_docking_records,
)
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


def test_protein_relative_path_from_ligand_filename_matches_targetdiff_rule():
    ligand_filename = 'FA11_HUMAN_388_625_0/4y8x_A_rec_4x6p_3yu_lig_tt_docked_6.sdf'
    assert protein_relative_path_from_ligand_filename(ligand_filename) == 'FA11_HUMAN_388_625_0/4y8x_A_rec.pdb'


def test_summarize_docking_records_vina_score_uses_successful_dockings_only():
    records = [
        DockingRecord(
            mode='vina_score',
            is_success=True,
            error=None,
            receptor_pdb_path='rec_a.pdb',
            receptor_pdbqt_path='rec_a.pdbqt',
            ligand_sdf_path='lig_a.sdf',
            ligand_pdbqt_path='lig_a.pdbqt',
            center_x=1.0,
            center_y=2.0,
            center_z=3.0,
            size_x=18.0,
            size_y=19.0,
            size_z=20.0,
            score_only_affinity=-8.5,
            minimize_affinity=-8.0,
            dock_affinity=None,
        ),
        DockingRecord(
            mode='vina_score',
            is_success=False,
            error='dock failed',
            receptor_pdb_path='rec_b.pdb',
            receptor_pdbqt_path='rec_b.pdbqt',
            ligand_sdf_path='lig_b.sdf',
            ligand_pdbqt_path='lig_b.pdbqt',
            center_x=1.0,
            center_y=2.0,
            center_z=3.0,
            size_x=18.0,
            size_y=19.0,
            size_z=20.0,
            score_only_affinity=99.9,
            minimize_affinity=99.9,
            dock_affinity=None,
        ),
    ]
    summary = summarize_docking_records(records)
    assert summary['num_docked'] == 1
    assert summary['docking_success_fraction'] == 0.5
    assert summary['vina_score_mean'] == -8.5
    assert summary['vina_score_median'] == -8.5
    assert summary['vina_min_mean'] == -8.0
    assert summary['vina_min_median'] == -8.0


def test_translate_ligand_to_center_matches_requested_center():
    mol = Chem.AddHs(Chem.MolFromSmiles('CCO'))
    assert AllChem.EmbedMolecule(mol, randomSeed=0) == 0
    target = np.asarray([10.0, -3.0, 5.5], dtype=np.float32)
    moved = _translate_ligand_to_center(mol, target)
    center = _ligand_center_of_mass(moved)
    assert np.allclose(center, target, atol=1e-3)
