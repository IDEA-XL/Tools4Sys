import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FAIL_SCORE = 99.9


@dataclass(frozen=True)
class DockingRecord:
    score: float
    is_success: bool
    error: str | None
    receptor_pdb_path: str
    receptor_pdbqt_path: str
    native_ligand_path: str
    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float


def summarize_docking_records(records):
    if not records:
        raise ValueError('records must be non-empty')
    scores = [float(record.score) for record in records]
    sorted_scores = sorted(scores)
    middle = len(sorted_scores) // 2
    if len(sorted_scores) % 2 == 1:
        median = sorted_scores[middle]
    else:
        median = 0.5 * (sorted_scores[middle - 1] + sorted_scores[middle])
    return {
        'docking_score_mean': float(sum(scores) / len(scores)),
        'docking_score_median': float(median),
        'docking_success_fraction': float(sum(1.0 if record.is_success else 0.0 for record in records) / len(records)),
        'num_docked': int(sum(1 for record in records if record.is_success)),
    }


def _require_command(name):
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f'Required command not found on PATH: {name}')
    return path


def _load_native_ligand_coords(native_ligand_path):
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(native_ligand_path), removeHs=False)
    if not supplier:
        raise RuntimeError(f'Failed to read native ligand SDF: {native_ligand_path}')
    mol = supplier[0]
    if mol is None:
        raise RuntimeError(f'Native ligand SDF contains no readable molecule: {native_ligand_path}')
    conformer = mol.GetConformer()
    coords = conformer.GetPositions()
    if coords is None or len(coords) == 0:
        raise RuntimeError(f'Native ligand has no 3D coordinates: {native_ligand_path}')
    return np.asarray(coords, dtype=np.float32)


def _resolve_box(native_ligand_path, box_padding, min_box_size):
    coords = _load_native_ligand_coords(native_ligand_path)
    center = coords.mean(axis=0)
    span = coords.max(axis=0) - coords.min(axis=0)
    size = np.maximum(span + float(box_padding), float(min_box_size))
    return center.astype(np.float32), size.astype(np.float32)


class CrossDockedDockingEvaluator:
    def __init__(
        self,
        crossdocked_root,
        qvina_path,
        cache_dir,
        exhaustiveness=1,
        num_cpu_dock=5,
        num_modes=10,
        timeout_gen3d=30,
        timeout_dock=100,
        box_padding=8.0,
        min_box_size=18.0,
    ):
        self.crossdocked_root = Path(crossdocked_root).expanduser().resolve()
        self.qvina_path = Path(qvina_path).expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.exhaustiveness = int(exhaustiveness)
        self.num_cpu_dock = int(num_cpu_dock)
        self.num_modes = int(num_modes)
        self.timeout_gen3d = int(timeout_gen3d)
        self.timeout_dock = int(timeout_dock)
        self.box_padding = float(box_padding)
        self.min_box_size = float(min_box_size)

        if not self.crossdocked_root.exists():
            raise FileNotFoundError(f'crossdocked_root not found: {self.crossdocked_root}')
        if not self.crossdocked_root.is_dir():
            raise NotADirectoryError(f'crossdocked_root is not a directory: {self.crossdocked_root}')
        if not self.qvina_path.exists():
            raise FileNotFoundError(f'qvina binary not found: {self.qvina_path}')
        if not os.access(self.qvina_path, os.X_OK):
            raise PermissionError(f'qvina binary is not executable: {self.qvina_path}')
        if self.exhaustiveness <= 0:
            raise ValueError('exhaustiveness must be positive')
        if self.num_cpu_dock <= 0:
            raise ValueError('num_cpu_dock must be positive')
        if self.num_modes <= 0:
            raise ValueError('num_modes must be positive')
        if self.box_padding <= 0:
            raise ValueError('box_padding must be positive')
        if self.min_box_size <= 0:
            raise ValueError('min_box_size must be positive')

        _require_command('obabel')
        try:
            from openbabel import pybel as _pybel  # noqa: F401
        except Exception as exc:
            raise RuntimeError('openbabel.pybel is required for docking evaluation') from exc

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def close(self):
        return None

    def _resolve_entry_paths(self, entry):
        receptor_pdb_path = self.crossdocked_root / str(entry['protein_filename'])
        native_ligand_path = self.crossdocked_root / str(entry['ligand_filename'])
        if not receptor_pdb_path.exists():
            raise FileNotFoundError(f'Receptor file not found: {receptor_pdb_path}')
        if not native_ligand_path.exists():
            raise FileNotFoundError(f'Native ligand file not found: {native_ligand_path}')
        return receptor_pdb_path, native_ligand_path

    def _cached_receptor_pdbqt_path(self, receptor_pdb_path):
        relative = receptor_pdb_path.relative_to(self.crossdocked_root)
        return self.cache_dir / relative.with_suffix('.pdbqt')

    def _ensure_receptor_pdbqt(self, receptor_pdb_path):
        receptor_pdbqt_path = self._cached_receptor_pdbqt_path(receptor_pdb_path)
        if receptor_pdbqt_path.exists():
            return receptor_pdbqt_path

        receptor_pdbqt_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            'obabel',
            str(receptor_pdb_path),
            '-O',
            str(receptor_pdbqt_path),
            '-xr',
        ]
        subprocess.check_output(
            command,
            stderr=subprocess.STDOUT,
            timeout=self.timeout_gen3d,
            universal_newlines=True,
        )
        if not receptor_pdbqt_path.exists():
            raise RuntimeError(f'Failed to create receptor pdbqt: {receptor_pdbqt_path}')
        return receptor_pdbqt_path

    def _generate_ligand_mol(self, smiles, ligand_mol_path):
        command = ['obabel', f'-:{smiles}', '--gen3D', '-O', str(ligand_mol_path)]
        subprocess.check_output(
            command,
            stderr=subprocess.STDOUT,
            timeout=self.timeout_gen3d,
            universal_newlines=True,
        )

    def _convert_ligand_to_pdbqt(self, ligand_mol_path, ligand_pdbqt_path):
        from openbabel import pybel

        molecules = list(pybel.readfile('mol', str(ligand_mol_path)))
        if not molecules:
            raise RuntimeError(f'Failed to read generated ligand mol file: {ligand_mol_path}')
        molecules[0].write('pdbqt', str(ligand_pdbqt_path), overwrite=True)

    def _dock_once(self, receptor_pdbqt_path, ligand_pdbqt_path, docking_pdbqt_path, center, size):
        command = [
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
        ]
        result = subprocess.check_output(
            command,
            stderr=subprocess.STDOUT,
            timeout=self.timeout_dock,
            universal_newlines=True,
        )
        affinity_list = []
        parsing = False
        for line in result.splitlines():
            if line.startswith('-----+'):
                parsing = True
                continue
            if not parsing:
                continue
            if line.startswith('Writing output') or line.startswith('Refine time'):
                break
            parts = line.strip().split()
            if not parts or not parts[0].isdigit():
                break
            affinity_list.append(float(parts[1]))
        if not affinity_list:
            raise RuntimeError('qvina did not return any affinity values')
        return float(affinity_list[0])

    def score(self, entries, smiles_list):
        if len(entries) != len(smiles_list):
            raise ValueError(f'entries and smiles_list length mismatch: {len(entries)} vs {len(smiles_list)}')

        records = []
        for entry, smiles in zip(entries, smiles_list):
            receptor_pdb_path, native_ligand_path = self._resolve_entry_paths(entry)
            center, size = _resolve_box(
                native_ligand_path=native_ligand_path,
                box_padding=self.box_padding,
                min_box_size=self.min_box_size,
            )
            receptor_pdbqt_path = self._ensure_receptor_pdbqt(receptor_pdb_path)

            if smiles is None:
                records.append(
                    DockingRecord(
                        score=FAIL_SCORE,
                        is_success=False,
                        error='Generated molecule is invalid; no SMILES available for docking',
                        receptor_pdb_path=str(receptor_pdb_path),
                        receptor_pdbqt_path=str(receptor_pdbqt_path),
                        native_ligand_path=str(native_ligand_path),
                        center_x=float(center[0]),
                        center_y=float(center[1]),
                        center_z=float(center[2]),
                        size_x=float(size[0]),
                        size_y=float(size[1]),
                        size_z=float(size[2]),
                    )
                )
                continue

            try:
                with tempfile.TemporaryDirectory(prefix='crossdocked_dock_') as tmpdir:
                    tmpdir_path = Path(tmpdir)
                    ligand_mol_path = tmpdir_path / 'ligand.mol'
                    ligand_pdbqt_path = tmpdir_path / 'ligand.pdbqt'
                    docking_pdbqt_path = tmpdir_path / 'dock_out.pdbqt'
                    self._generate_ligand_mol(smiles, ligand_mol_path)
                    self._convert_ligand_to_pdbqt(ligand_mol_path, ligand_pdbqt_path)
                    score = self._dock_once(receptor_pdbqt_path, ligand_pdbqt_path, docking_pdbqt_path, center, size)
                records.append(
                    DockingRecord(
                        score=score,
                        is_success=True,
                        error=None,
                        receptor_pdb_path=str(receptor_pdb_path),
                        receptor_pdbqt_path=str(receptor_pdbqt_path),
                        native_ligand_path=str(native_ligand_path),
                        center_x=float(center[0]),
                        center_y=float(center[1]),
                        center_z=float(center[2]),
                        size_x=float(size[0]),
                        size_y=float(size[1]),
                        size_z=float(size[2]),
                    )
                )
            except Exception as exc:
                records.append(
                    DockingRecord(
                        score=FAIL_SCORE,
                        is_success=False,
                        error=str(exc),
                        receptor_pdb_path=str(receptor_pdb_path),
                        receptor_pdbqt_path=str(receptor_pdbqt_path),
                        native_ligand_path=str(native_ligand_path),
                        center_x=float(center[0]),
                        center_y=float(center[1]),
                        center_z=float(center[2]),
                        size_x=float(size[0]),
                        size_y=float(size[1]),
                        size_z=float(size[2]),
                    )
                )
        return records
