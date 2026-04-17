import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FAIL_AFFINITY = 99.9
SUPPORTED_DOCKING_MODES = ('qvina', 'vina_score', 'vina_dock')


@dataclass(frozen=True)
class DockingRecord:
    mode: str
    is_success: bool
    error: str | None
    receptor_pdb_path: str
    receptor_pdbqt_path: str | None
    ligand_sdf_path: str | None
    ligand_pdbqt_path: str | None
    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float
    score_only_affinity: float | None
    minimize_affinity: float | None
    dock_affinity: float | None


def _median(values):
    sorted_values = sorted(float(value) for value in values)
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return float(sorted_values[middle])
    return float(0.5 * (sorted_values[middle - 1] + sorted_values[middle]))


def protein_relative_path_from_ligand_filename(ligand_filename):
    ligand_filename = str(ligand_filename)
    ligand_basename = os.path.basename(ligand_filename)
    if len(ligand_basename) < 10:
        raise ValueError(f'ligand_filename basename must have at least 10 characters, got {ligand_basename!r}')
    return os.path.join(os.path.dirname(ligand_filename), ligand_basename[:10] + '.pdb')


def summarize_docking_records(records):
    if not records:
        raise ValueError('records must be non-empty')
    modes = {record.mode for record in records}
    if len(modes) != 1:
        raise ValueError(f'Expected a single docking mode per summary, got {sorted(modes)}')
    mode = next(iter(modes))
    successful_records = [record for record in records if record.is_success]
    summary = {
        'docking_mode': mode,
        'docking_success_fraction': float(sum(1.0 if record.is_success else 0.0 for record in records) / len(records)),
        'num_docked': int(len(successful_records)),
    }
    if mode == 'qvina':
        scores = [record.dock_affinity for record in successful_records]
        if any(score is None for score in scores):
            raise ValueError('qvina records must populate dock_affinity')
        summary.update(
            {
                'qvina_mean': float('nan') if not scores else float(sum(float(score) for score in scores) / len(scores)),
                'qvina_median': float('nan') if not scores else _median(scores),
            }
        )
        return summary

    score_only = [record.score_only_affinity for record in successful_records]
    minimize = [record.minimize_affinity for record in successful_records]
    if any(score is None for score in score_only):
        raise ValueError(f'{mode} records must populate score_only_affinity')
    if any(score is None for score in minimize):
        raise ValueError(f'{mode} records must populate minimize_affinity')
    summary.update(
        {
            'vina_score_mean': float('nan') if not score_only else float(sum(float(score) for score in score_only) / len(score_only)),
            'vina_score_median': float('nan') if not score_only else _median(score_only),
            'vina_min_mean': float('nan') if not minimize else float(sum(float(score) for score in minimize) / len(minimize)),
            'vina_min_median': float('nan') if not minimize else _median(minimize),
        }
    )
    if mode == 'vina_dock':
        dock = [record.dock_affinity for record in successful_records]
        if any(score is None for score in dock):
            raise ValueError('vina_dock records must populate dock_affinity')
        summary.update(
            {
                'vina_dock_mean': float('nan') if not dock else float(sum(float(score) for score in dock) / len(dock)),
                'vina_dock_median': float('nan') if not dock else _median(dock),
            }
        )
    return summary


def _require_command(name):
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f'Required command not found on PATH: {name}')
    return path


def _embed_ligand_from_smiles(smiles):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'RDKit failed to parse generated SMILES: {smiles!r}')
    mol = Chem.AddHs(mol, addCoords=True)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise RuntimeError(f'RDKit failed to generate a 3D conformer for generated SMILES: {smiles!r}')
    return mol


def _write_ligand_sdf(ligand_rdmol, ligand_sdf_path):
    from rdkit import Chem

    writer = Chem.SDWriter(str(ligand_sdf_path))
    try:
        writer.write(ligand_rdmol)
    finally:
        writer.close()


def _ligand_center_and_size(ligand_rdmol, size_factor, buffer):
    pos = np.asarray(ligand_rdmol.GetConformer(0).GetPositions(), dtype=np.float32)
    center = (pos.max(axis=0) + pos.min(axis=0)) / 2.0
    if size_factor is None:
        size = np.asarray([20.0, 20.0, 20.0], dtype=np.float32)
    else:
        size = (pos.max(axis=0) - pos.min(axis=0)) * float(size_factor) + float(buffer)
    return center.astype(np.float32), size.astype(np.float32)


def _load_native_ligand_sdf(ligand_sdf_path):
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(ligand_sdf_path), removeHs=False)
    molecules = [mol for mol in supplier if mol is not None]
    if not molecules:
        raise RuntimeError(f'Failed to read native ligand SDF: {ligand_sdf_path}')
    return molecules[0]


class CrossDockedDockingEvaluator:
    def __init__(
        self,
        crossdocked_root,
        docking_mode='vina_score',
        cache_dir=None,
        qvina_path=None,
        exhaustiveness=8,
        num_cpu_dock=5,
        num_modes=10,
        timeout_gen3d=30,
        timeout_dock=100,
        size_factor=1.0,
        buffer=5.0,
    ):
        self.crossdocked_root = Path(crossdocked_root).expanduser().resolve()
        self.docking_mode = str(docking_mode)
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.qvina_path = None if qvina_path is None else Path(qvina_path).expanduser().resolve()
        self.exhaustiveness = int(exhaustiveness)
        self.num_cpu_dock = int(num_cpu_dock)
        self.num_modes = int(num_modes)
        self.timeout_gen3d = int(timeout_gen3d)
        self.timeout_dock = int(timeout_dock)
        self.size_factor = None if size_factor is None else float(size_factor)
        self.buffer = float(buffer)

        if not self.crossdocked_root.exists():
            raise FileNotFoundError(f'crossdocked_root not found: {self.crossdocked_root}')
        if not self.crossdocked_root.is_dir():
            raise NotADirectoryError(f'crossdocked_root is not a directory: {self.crossdocked_root}')
        if self.docking_mode not in SUPPORTED_DOCKING_MODES:
            raise ValueError(f'docking_mode must be one of {SUPPORTED_DOCKING_MODES}, got {self.docking_mode!r}')
        if self.exhaustiveness <= 0:
            raise ValueError('exhaustiveness must be positive')
        if self.num_cpu_dock <= 0:
            raise ValueError('num_cpu_dock must be positive')
        if self.num_modes <= 0:
            raise ValueError('num_modes must be positive')
        if self.size_factor is not None and self.size_factor <= 0:
            raise ValueError('size_factor must be positive when provided')
        if self.buffer < 0:
            raise ValueError('buffer must be non-negative')

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        _require_command('obabel')

        try:
            from openbabel import pybel as _pybel  # noqa: F401
        except Exception as exc:
            raise RuntimeError('Docking evaluation requires openbabel.pybel') from exc

        if self.docking_mode == 'qvina':
            if self.qvina_path is None:
                raise ValueError('qvina_path is required when docking_mode=qvina')
            if not self.qvina_path.exists():
                raise FileNotFoundError(f'qvina binary not found: {self.qvina_path}')
            if not os.access(self.qvina_path, os.X_OK):
                raise PermissionError(f'qvina binary is not executable: {self.qvina_path}')
        else:
            _require_command('pdb2pqr30')
            try:
                import AutoDockTools as _autodocktools  # noqa: F401
                import meeko as _meeko  # noqa: F401
                import vina as _vina  # noqa: F401
            except Exception as exc:
                raise RuntimeError(
                    'vina-based docking evaluation requires python packages vina, meeko, and AutoDockTools'
                ) from exc

    def close(self):
        return None

    def _resolve_receptor_pdb_path(self, entry):
        if 'ligand_filename' not in entry or entry['ligand_filename'] is None:
            raise ValueError('Manifest entry is missing ligand_filename; docking requires ligand_filename')
        relative_protein_path = protein_relative_path_from_ligand_filename(entry['ligand_filename'])
        receptor_pdb_path = self.crossdocked_root / relative_protein_path
        if not receptor_pdb_path.exists():
            raise FileNotFoundError(f'Receptor file not found: {receptor_pdb_path}')
        return receptor_pdb_path

    def _resolve_native_ligand_sdf_path(self, entry):
        if 'ligand_filename' not in entry or entry['ligand_filename'] is None:
            raise ValueError('Manifest entry is missing ligand_filename; docking requires ligand_filename')
        ligand_sdf_path = self.crossdocked_root / str(entry['ligand_filename'])
        if not ligand_sdf_path.exists():
            raise FileNotFoundError(f'Native ligand file not found: {ligand_sdf_path}')
        return ligand_sdf_path

    def _cached_receptor_path(self, receptor_pdb_path, suffix):
        relative = receptor_pdb_path.relative_to(self.crossdocked_root)
        return self.cache_dir / relative.with_suffix(suffix)

    def _ensure_qvina_receptor_pdbqt(self, receptor_pdb_path):
        receptor_pdbqt_path = self._cached_receptor_path(receptor_pdb_path, '.pdbqt')
        if receptor_pdbqt_path.exists():
            return receptor_pdbqt_path
        receptor_pdbqt_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_output(
            ['obabel', str(receptor_pdb_path), '-O', str(receptor_pdbqt_path), '-xr'],
            stderr=subprocess.STDOUT,
            timeout=self.timeout_gen3d,
            universal_newlines=True,
        )
        if not receptor_pdbqt_path.exists():
            raise RuntimeError(f'Failed to create receptor pdbqt: {receptor_pdbqt_path}')
        return receptor_pdbqt_path

    def _prepare_vina_receptor(self, receptor_pdb_path):
        import AutoDockTools

        receptor_pqr_path = self._cached_receptor_path(receptor_pdb_path, '.pqr')
        receptor_pdbqt_path = self._cached_receptor_path(receptor_pdb_path, '.pdbqt')
        receptor_pqr_path.parent.mkdir(parents=True, exist_ok=True)

        if not receptor_pqr_path.exists():
            subprocess.check_output(
                ['pdb2pqr30', '--ff=AMBER', str(receptor_pdb_path), str(receptor_pqr_path)],
                stderr=subprocess.STDOUT,
                timeout=self.timeout_dock,
                universal_newlines=True,
            )
        if not receptor_pdbqt_path.exists():
            prepare_receptor = Path(AutoDockTools.__path__[0]) / 'Utilities24' / 'prepare_receptor4.py'
            if not prepare_receptor.exists():
                raise FileNotFoundError(f'prepare_receptor4.py not found under AutoDockTools: {prepare_receptor}')
            subprocess.check_output(
                [sys.executable, str(prepare_receptor), '-r', str(receptor_pqr_path), '-o', str(receptor_pdbqt_path)],
                stderr=subprocess.STDOUT,
                timeout=self.timeout_dock,
                universal_newlines=True,
            )
        if not receptor_pdbqt_path.exists():
            raise RuntimeError(f'Failed to create receptor pdbqt: {receptor_pdbqt_path}')
        return receptor_pdbqt_path

    def _convert_ligand_to_qvina_pdbqt(self, ligand_sdf_path, ligand_pdbqt_path):
        from openbabel import pybel

        molecules = list(pybel.readfile('sdf', str(ligand_sdf_path)))
        if not molecules:
            raise RuntimeError(f'Failed to read generated ligand SDF: {ligand_sdf_path}')
        molecules[0].write('pdbqt', str(ligand_pdbqt_path), overwrite=True)

    def _convert_ligand_to_vina_pdbqt(self, ligand_sdf_path, ligand_pdbqt_path):
        from rdkit import Chem
        from meeko import MoleculePreparation

        supplier = Chem.SDMolSupplier(str(ligand_sdf_path), removeHs=False)
        molecules = [mol for mol in supplier if mol is not None]
        if not molecules:
            raise RuntimeError(f'Failed to read generated ligand SDF: {ligand_sdf_path}')
        preparator = MoleculePreparation()
        preparator.prepare(molecules[0])
        preparator.write_pdbqt_file(str(ligand_pdbqt_path))
        if not ligand_pdbqt_path.exists():
            raise RuntimeError(f'Failed to create ligand pdbqt: {ligand_pdbqt_path}')

    def _run_qvina(self, receptor_pdbqt_path, ligand_pdbqt_path, center, size, docking_pdbqt_path):
        output = subprocess.check_output(
            [
                str(self.qvina_path),
                '--receptor',
                str(receptor_pdbqt_path),
                '--ligand',
                str(ligand_pdbqt_path),
                '--out',
                str(docking_pdbqt_path),
                '--center_x',
                str(float(center[0])),
                '--center_y',
                str(float(center[1])),
                '--center_z',
                str(float(center[2])),
                '--size_x',
                str(float(size[0])),
                '--size_y',
                str(float(size[1])),
                '--size_z',
                str(float(size[2])),
                '--cpu',
                str(self.num_cpu_dock),
                '--num_modes',
                str(self.num_modes),
                '--exhaustiveness',
                str(self.exhaustiveness),
            ],
            stderr=subprocess.STDOUT,
            timeout=self.timeout_dock,
            universal_newlines=True,
        )
        affinity_values = []
        parsing = False
        for line in output.splitlines():
            if line.startswith('-----+'):
                parsing = True
                continue
            if not parsing:
                continue
            parts = line.strip().split()
            if not parts or not parts[0].isdigit():
                break
            affinity_values.append(float(parts[1]))
        if not affinity_values:
            raise RuntimeError('qvina did not return any affinity values')
        return float(affinity_values[0])

    def _run_vina(self, receptor_pdbqt_path, ligand_pdbqt_path, center, size):
        from vina import Vina

        vina_task = Vina(sf_name='vina', seed=0, verbosity=0)
        vina_task.set_receptor(str(receptor_pdbqt_path))
        vina_task.set_ligand_from_file(str(ligand_pdbqt_path))
        vina_task.compute_vina_maps(
            center=[float(center[0]), float(center[1]), float(center[2])],
            box_size=[float(size[0]), float(size[1]), float(size[2])],
        )
        score_only = float(vina_task.score()[0])
        minimize = float(vina_task.optimize()[0])
        dock = None
        if self.docking_mode == 'vina_dock':
            vina_task.dock(exhaustiveness=self.exhaustiveness, n_poses=1)
            dock = float(vina_task.energies(n_poses=1)[0][0])
        return score_only, minimize, dock

    def _failure_record(
        self,
        *,
        receptor_pdb_path,
        receptor_pdbqt_path,
        ligand_sdf_path,
        ligand_pdbqt_path,
        center,
        size,
        error,
    ):
        score_only = None
        minimize = None
        dock = None
        if self.docking_mode == 'qvina':
            dock = FAIL_AFFINITY
        else:
            score_only = FAIL_AFFINITY
            minimize = FAIL_AFFINITY
            if self.docking_mode == 'vina_dock':
                dock = FAIL_AFFINITY
        return DockingRecord(
            mode=self.docking_mode,
            is_success=False,
            error=str(error),
            receptor_pdb_path='' if receptor_pdb_path is None else str(receptor_pdb_path),
            receptor_pdbqt_path=None if receptor_pdbqt_path is None else str(receptor_pdbqt_path),
            ligand_sdf_path=None if ligand_sdf_path is None else str(ligand_sdf_path),
            ligand_pdbqt_path=None if ligand_pdbqt_path is None else str(ligand_pdbqt_path),
            center_x=float(center[0]),
            center_y=float(center[1]),
            center_z=float(center[2]),
            size_x=float(size[0]),
            size_y=float(size[1]),
            size_z=float(size[2]),
            score_only_affinity=score_only,
            minimize_affinity=minimize,
            dock_affinity=dock,
        )

    def score(self, entries, smiles_list):
        if len(entries) != len(smiles_list):
            raise ValueError(f'entries and smiles_list length mismatch: {len(entries)} vs {len(smiles_list)}')

        records = []
        for entry, smiles in zip(entries, smiles_list):
            receptor_pdb_path = None
            native_ligand_sdf_path = None
            receptor_pdbqt_path = None
            ligand_sdf_path = None
            ligand_pdbqt_path = None
            center = np.asarray([float('nan'), float('nan'), float('nan')], dtype=np.float32)
            size = np.asarray([float('nan'), float('nan'), float('nan')], dtype=np.float32)

            if smiles is None:
                records.append(
                    self._failure_record(
                        receptor_pdb_path=receptor_pdb_path,
                        receptor_pdbqt_path=receptor_pdbqt_path,
                        ligand_sdf_path=ligand_sdf_path,
                        ligand_pdbqt_path=ligand_pdbqt_path,
                        center=center,
                        size=size,
                        error='Generated molecule is invalid; no SMILES available for docking',
                    )
                )
                continue

            try:
                receptor_pdb_path = self._resolve_receptor_pdb_path(entry)
                native_ligand_sdf_path = self._resolve_native_ligand_sdf_path(entry)
                ligand_rdmol = _embed_ligand_from_smiles(smiles)
                native_ligand_rdmol = _load_native_ligand_sdf(native_ligand_sdf_path)
                if self.docking_mode == 'qvina':
                    center, size = _ligand_center_and_size(native_ligand_rdmol, self.size_factor, 0.0)
                else:
                    center, size = _ligand_center_and_size(native_ligand_rdmol, self.size_factor, self.buffer)

                with tempfile.TemporaryDirectory(prefix='crossdocked_dock_') as tmpdir:
                    tmpdir_path = Path(tmpdir)
                    ligand_sdf_path = tmpdir_path / 'ligand.sdf'
                    ligand_pdbqt_path = tmpdir_path / 'ligand.pdbqt'
                    _write_ligand_sdf(ligand_rdmol, ligand_sdf_path)

                    if self.docking_mode == 'qvina':
                        receptor_pdbqt_path = self._ensure_qvina_receptor_pdbqt(receptor_pdb_path)
                        self._convert_ligand_to_qvina_pdbqt(ligand_sdf_path, ligand_pdbqt_path)
                        docking_pdbqt_path = tmpdir_path / 'dock_out.pdbqt'
                        dock_affinity = self._run_qvina(
                            receptor_pdbqt_path=receptor_pdbqt_path,
                            ligand_pdbqt_path=ligand_pdbqt_path,
                            center=center,
                            size=size,
                            docking_pdbqt_path=docking_pdbqt_path,
                        )
                        records.append(
                            DockingRecord(
                                mode=self.docking_mode,
                                is_success=True,
                                error=None,
                                receptor_pdb_path=str(receptor_pdb_path),
                                receptor_pdbqt_path=str(receptor_pdbqt_path),
                                ligand_sdf_path=str(ligand_sdf_path),
                                ligand_pdbqt_path=str(ligand_pdbqt_path),
                                center_x=float(center[0]),
                                center_y=float(center[1]),
                                center_z=float(center[2]),
                                size_x=float(size[0]),
                                size_y=float(size[1]),
                                size_z=float(size[2]),
                                score_only_affinity=None,
                                minimize_affinity=None,
                                dock_affinity=dock_affinity,
                            )
                        )
                    else:
                        receptor_pdbqt_path = self._prepare_vina_receptor(receptor_pdb_path)
                        self._convert_ligand_to_vina_pdbqt(ligand_sdf_path, ligand_pdbqt_path)
                        score_only, minimize, dock_affinity = self._run_vina(
                            receptor_pdbqt_path=receptor_pdbqt_path,
                            ligand_pdbqt_path=ligand_pdbqt_path,
                            center=center,
                            size=size,
                        )
                        records.append(
                            DockingRecord(
                                mode=self.docking_mode,
                                is_success=True,
                                error=None,
                                receptor_pdb_path=str(receptor_pdb_path),
                                receptor_pdbqt_path=str(receptor_pdbqt_path),
                                ligand_sdf_path=str(ligand_sdf_path),
                                ligand_pdbqt_path=str(ligand_pdbqt_path),
                                center_x=float(center[0]),
                                center_y=float(center[1]),
                                center_z=float(center[2]),
                                size_x=float(size[0]),
                                size_y=float(size[1]),
                                size_z=float(size[2]),
                                score_only_affinity=score_only,
                                minimize_affinity=minimize,
                                dock_affinity=dock_affinity,
                            )
                        )
            except Exception as exc:
                records.append(
                    self._failure_record(
                        receptor_pdb_path=receptor_pdb_path,
                        receptor_pdbqt_path=receptor_pdbqt_path,
                        ligand_sdf_path=ligand_sdf_path,
                        ligand_pdbqt_path=ligand_pdbqt_path,
                        center=center,
                        size=size,
                        error=exc,
                    )
                )
        return records
