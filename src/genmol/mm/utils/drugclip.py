from __future__ import annotations

import hashlib
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from genmol.mm.crossdocked import _open_lmdb


@dataclass(frozen=True)
class DrugCLIPConfig:
    checkpoint_path: str
    crossdocked_lmdb_path: str
    device: str
    batch_size: int = 64
    max_pocket_atoms: int = 256
    num_conformers: int = 1
    conformer_num_workers: int = 1
    use_fp16: bool = True
    seed: int = 1
    mol_cache_size: int = 4096
    pocket_cache_size: int = 4096


@dataclass(frozen=True)
class _DrugCLIPFeatures:
    tokens: torch.Tensor
    distance: torch.Tensor
    edge_type: torch.Tensor


@dataclass(frozen=True)
class _PreparedMoleculeFeatureResult:
    feature: _DrugCLIPFeatures | None
    failure_reason: str | None


class _DrugCLIPDictionary:
    def __init__(self, tokens: list[str]):
        if not tokens:
            raise ValueError('DrugCLIP dictionary must be non-empty')
        self._tokens = list(tokens)
        self._indices = {token: idx for idx, token in enumerate(self._tokens)}
        for required_token in ('[PAD]', '[CLS]', '[SEP]', '[UNK]'):
            if required_token not in self._indices:
                raise ValueError(f'DrugCLIP dictionary is missing required token {required_token!r}')

    @classmethod
    def load(cls, path: Path):
        with open(path) as handle:
            tokens = [line.strip() for line in handle if line.strip()]
        return cls(tokens)

    def __len__(self) -> int:
        return len(self._tokens)

    def pad(self) -> int:
        return self._indices['[PAD]']

    def bos(self) -> int:
        return self._indices['[CLS]']

    def eos(self) -> int:
        return self._indices['[SEP]']

    def unk(self) -> int:
        return self._indices['[UNK]']

    def encode(self, token: str) -> int:
        return self._indices.get(token, self.unk())

    def add_symbol(self, token: str) -> int:
        existing = self._indices.get(token)
        if existing is not None:
            return existing
        index = len(self._tokens)
        self._tokens.append(token)
        self._indices[token] = index
        return index


class _CrossDockedRawEntryStore:
    def __init__(self, lmdb_path: str):
        self._lmdb_path = lmdb_path
        self._env = _open_lmdb(lmdb_path)
        self._source_index_to_lmdb_key = None
        self._entry_cache: OrderedDict[int, dict] = OrderedDict()
        self._entry_cache_size = 1024

    def close(self):
        self._env.close()

    def _build_source_index_to_lmdb_key(self):
        mapping = {}
        with self._env.begin(write=False) as txn:
            cursor = txn.cursor()
            source_index = 0
            for raw_key, _payload in cursor:
                key = raw_key.decode('utf-8')
                if key == 'num_examples':
                    continue
                if not key.isdigit():
                    raise ValueError(f'CrossDocked LMDB contains a non-numeric sample key: {key!r}')
                mapping[source_index] = int(key)
                source_index += 1
        if not mapping:
            raise ValueError(f'CrossDocked LMDB contains no numeric sample keys: {self._lmdb_path}')
        self._source_index_to_lmdb_key = mapping

    def _resolve_lmdb_key(self, pocket_entry: dict) -> int:
        raw_lmdb_key = pocket_entry.get('lmdb_key')
        if raw_lmdb_key is not None:
            return int(raw_lmdb_key)
        if 'source_index' not in pocket_entry:
            raise ValueError('DrugCLIP scoring requires manifest entries with source_index or lmdb_key')
        if self._source_index_to_lmdb_key is None:
            self._build_source_index_to_lmdb_key()
        source_index = int(pocket_entry['source_index'])
        try:
            return int(self._source_index_to_lmdb_key[source_index])
        except KeyError as exc:
            raise KeyError(f'Failed to resolve CrossDocked source_index {source_index} to an LMDB key') from exc

    def get_entry(self, pocket_entry: dict) -> dict:
        lmdb_key = self._resolve_lmdb_key(pocket_entry)
        cached = self._entry_cache.get(lmdb_key)
        if cached is not None:
            self._entry_cache.move_to_end(lmdb_key)
            return cached
        with self._env.begin(write=False) as txn:
            payload = txn.get(str(lmdb_key).encode('utf-8'))
        if payload is None:
            raise KeyError(f'CrossDocked LMDB key {lmdb_key} not found in {self._lmdb_path}')
        entry = pickle.loads(payload)
        self._entry_cache[lmdb_key] = entry
        self._entry_cache.move_to_end(lmdb_key)
        while len(self._entry_cache) > self._entry_cache_size:
            self._entry_cache.popitem(last=False)
        return entry


def _stable_uint32(text: str) -> int:
    digest = hashlib.sha256(text.encode('utf-8')).digest()
    return int.from_bytes(digest[:4], byteorder='little', signed=False)


def _decode_atom_name(atom_name) -> str:
    if isinstance(atom_name, bytes):
        return atom_name.decode('utf-8').strip()
    return str(atom_name).strip()


def _normalize_pocket_atom_label(atom_name: str) -> str:
    if not atom_name:
        raise ValueError('Encountered an empty protein_atom_name while preparing DrugCLIP pocket atoms')
    if atom_name[0].isdigit():
        if len(atom_name) < 2:
            raise ValueError(f'Invalid protein_atom_name for DrugCLIP pocket atom mapping: {atom_name!r}')
        return atom_name[1]
    return atom_name[0]


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponent = np.exp(shifted)
    denominator = np.sum(exponent)
    if denominator <= 0.0:
        raise ValueError('DrugCLIP pocket cropping encountered a non-positive softmax denominator')
    return exponent / denominator


def _prepare_pocket_features(raw_entry: dict, *, max_pocket_atoms: int, seed: int, source_index: int, dictionary: _DrugCLIPDictionary):
    if 'protein_atom_name' not in raw_entry or 'protein_pos' not in raw_entry:
        missing = sorted({'protein_atom_name', 'protein_pos'}.difference(raw_entry.keys()))
        raise ValueError(
            f'DrugCLIP pocket scoring requires atom-level CrossDocked protein fields, missing {missing}'
        )

    atom_names = [_decode_atom_name(atom_name) for atom_name in raw_entry['protein_atom_name']]
    coordinates = np.asarray(raw_entry['protein_pos'], dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(
            'CrossDocked protein_pos must have shape [num_atoms, 3] for DrugCLIP scoring, '
            f'got {list(coordinates.shape)}'
        )
    if len(atom_names) != coordinates.shape[0]:
        raise ValueError(
            'CrossDocked protein_atom_name and protein_pos must share the same length for DrugCLIP scoring'
        )

    pocket_atoms = np.asarray([_normalize_pocket_atom_label(atom_name) for atom_name in atom_names], dtype=object)
    non_hydrogen_mask = pocket_atoms != 'H'
    pocket_atoms = pocket_atoms[non_hydrogen_mask]
    coordinates = coordinates[non_hydrogen_mask]
    if pocket_atoms.size == 0:
        raise ValueError('DrugCLIP pocket preprocessing removed every pocket atom as hydrogen')

    if max_pocket_atoms > 0 and pocket_atoms.size > max_pocket_atoms:
        distance = np.linalg.norm(coordinates - coordinates.mean(axis=0), axis=1) + 1.0
        weight = _softmax(np.reciprocal(distance))
        rng = np.random.default_rng(seed + int(source_index))
        selected = rng.choice(pocket_atoms.size, size=max_pocket_atoms, replace=False, p=weight)
        pocket_atoms = pocket_atoms[selected]
        coordinates = coordinates[selected]

    coordinates = coordinates - coordinates.mean(axis=0)
    return _build_features(pocket_atoms.tolist(), coordinates, dictionary)


def _prepare_molecule_features(
    smiles: str,
    *,
    num_conformers: int,
    conformer_num_workers: int,
    seed: int,
    dictionary: _DrugCLIPDictionary,
):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return _PreparedMoleculeFeatureResult(feature=None, failure_reason='smiles_parse')

    try:
        molecule = Chem.AddHs(molecule)
        AllChem.EmbedMultipleConfs(
            molecule,
            numConfs=int(num_conformers),
            numThreads=int(conformer_num_workers),
            randomSeed=int(seed),
            pruneRmsThresh=1.0,
            maxAttempts=10000,
            useRandomCoords=False,
        )
        try:
            AllChem.MMFFOptimizeMoleculeConfs(molecule, numThreads=int(conformer_num_workers))
        except Exception:
            pass
        molecule = Chem.RemoveHs(molecule)
    except Exception:
        return _PreparedMoleculeFeatureResult(feature=None, failure_reason='embed_exception')

    if molecule.GetNumConformers() <= 0:
        return _PreparedMoleculeFeatureResult(feature=None, failure_reason='zero_conformer')

    coordinates = np.asarray(molecule.GetConformer(0).GetPositions(), dtype=np.float32)
    atoms = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    if len(atoms) != coordinates.shape[0]:
        raise ValueError('DrugCLIP molecule atom labels and coordinates must have matching lengths')
    if not atoms:
        return _PreparedMoleculeFeatureResult(feature=None, failure_reason='empty_atom_list')

    coordinates = coordinates - coordinates.mean(axis=0)
    return _PreparedMoleculeFeatureResult(
        feature=_build_features(atoms, coordinates, dictionary),
        failure_reason=None,
    )


def _build_features(atom_tokens: list[str], coordinates: np.ndarray, dictionary: _DrugCLIPDictionary):
    if not atom_tokens:
        raise ValueError('DrugCLIP features require at least one atom token')
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f'DrugCLIP coordinates must have shape [num_atoms, 3], got {list(coordinates.shape)}')
    if len(atom_tokens) != coordinates.shape[0]:
        raise ValueError('DrugCLIP atom token count must match coordinate count')

    token_ids = [dictionary.bos()]
    token_ids.extend(dictionary.encode(token) for token in atom_tokens)
    token_ids.append(dictionary.eos())

    padded_coordinates = np.zeros((coordinates.shape[0] + 2, 3), dtype=np.float32)
    padded_coordinates[1:-1] = coordinates.astype(np.float32)

    tokens = torch.as_tensor(token_ids, dtype=torch.long)
    coords_tensor = torch.as_tensor(padded_coordinates, dtype=torch.float32)
    distance = torch.cdist(coords_tensor, coords_tensor)
    edge_type = tokens.view(-1, 1) * len(dictionary) + tokens.view(1, -1)
    return _DrugCLIPFeatures(tokens=tokens, distance=distance, edge_type=edge_type)


def _collate_features(features: list[_DrugCLIPFeatures], pad_idx: int, device: torch.device):
    if not features:
        raise ValueError('DrugCLIP collation requires a non-empty feature batch')
    max_len = max(int(feature.tokens.numel()) for feature in features)
    batch_size = len(features)
    token_batch = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)
    distance_batch = torch.zeros((batch_size, max_len, max_len), dtype=torch.float32)
    edge_type_batch = torch.zeros((batch_size, max_len, max_len), dtype=torch.long)
    for batch_index, feature in enumerate(features):
        item_len = int(feature.tokens.numel())
        token_batch[batch_index, :item_len] = feature.tokens
        distance_batch[batch_index, :item_len, :item_len] = feature.distance
        edge_type_batch[batch_index, :item_len, :item_len] = feature.edge_type
    return (
        token_batch.to(device=device, non_blocking=False),
        distance_batch.to(device=device, non_blocking=False),
        edge_type_batch.to(device=device, non_blocking=False),
    )


class DrugCLIPScorer:
    def __init__(self, config: DrugCLIPConfig):
        self.config = config
        self.device = torch.device(config.device)
        if config.batch_size <= 0:
            raise ValueError(f'DrugCLIP batch_size must be positive, got {config.batch_size}')
        if config.max_pocket_atoms <= 0:
            raise ValueError(f'DrugCLIP max_pocket_atoms must be positive, got {config.max_pocket_atoms}')
        if config.num_conformers <= 0:
            raise ValueError(f'DrugCLIP num_conformers must be positive, got {config.num_conformers}')
        if config.conformer_num_workers <= 0:
            raise ValueError(
                f'DrugCLIP conformer_num_workers must be positive, got {config.conformer_num_workers}'
            )
        if config.mol_cache_size <= 0:
            raise ValueError(f'DrugCLIP mol_cache_size must be positive, got {config.mol_cache_size}')
        if config.pocket_cache_size <= 0:
            raise ValueError(f'DrugCLIP pocket_cache_size must be positive, got {config.pocket_cache_size}')

        vendor_root = Path(__file__).resolve().parent / 'drugclip_vendor'
        from .drugclip_vendor import DrugCLIPEncoderModel, build_default_drugclip_args

        self.mol_dictionary = _DrugCLIPDictionary.load(vendor_root / 'dict_mol.txt')
        self.pocket_dictionary = _DrugCLIPDictionary.load(vendor_root / 'dict_pkt.txt')
        self.mol_dictionary.add_symbol('[MASK]')
        self.pocket_dictionary.add_symbol('[MASK]')

        args = build_default_drugclip_args()
        self.model = DrugCLIPEncoderModel(
            args,
            mol_vocab_size=len(self.mol_dictionary),
            pocket_vocab_size=len(self.pocket_dictionary),
            mol_padding_idx=self.mol_dictionary.pad(),
            pocket_padding_idx=self.pocket_dictionary.pad(),
        )
        checkpoint = torch.load(config.checkpoint_path, map_location='cpu', weights_only=False)
        if not isinstance(checkpoint, dict) or 'model' not in checkpoint:
            raise ValueError(
                'DrugCLIP checkpoint must be a dict payload containing a "model" state dict'
            )
        incompatible = self.model.load_state_dict(checkpoint['model'], strict=False)
        if incompatible.missing_keys:
            raise ValueError(f'DrugCLIP checkpoint is missing required parameters: {sorted(incompatible.missing_keys)}')

        self.model.requires_grad_(False)
        self.model.eval()
        self.model.to(self.device)
        if config.use_fp16 and self.device.type == 'cuda':
            self.model.half()

        self.raw_entry_store = _CrossDockedRawEntryStore(config.crossdocked_lmdb_path)
        self._mol_embedding_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._pocket_embedding_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.last_score_stats = self._empty_last_score_stats()

    @staticmethod
    def _empty_last_score_stats():
        return {
            'drugclip_input_count': 0,
            'drugclip_unique_smiles_count': 0,
            'drugclip_unique_pocket_count': 0,
            'drugclip_molecule_cache_hit_count': 0,
            'drugclip_molecule_cache_miss_count': 0,
            'drugclip_unique_smiles_success_count': 0,
            'drugclip_unique_smiles_failure_count': 0,
            'drugclip_score_success_count': 0,
            'drugclip_score_failure_count': 0,
            'drugclip_score_success_fraction': 0.0,
            'drugclip_fail_smiles_parse_count': 0,
            'drugclip_fail_embed_exception_count': 0,
            'drugclip_fail_zero_conformer_count': 0,
            'drugclip_fail_empty_atom_list_count': 0,
        }

    def close(self):
        self.raw_entry_store.close()

    def _get_from_cache(self, cache: OrderedDict, key):
        tensor = cache.get(key)
        if tensor is None:
            return None
        cache.move_to_end(key)
        return tensor

    def _put_in_cache(self, cache: OrderedDict, key, tensor: torch.Tensor, max_size: int):
        cache[key] = tensor.detach()
        cache.move_to_end(key)
        while len(cache) > max_size:
            cache.popitem(last=False)

    def _encode_missing_molecules(self, smiles_list: list[str]) -> tuple[dict[str, torch.Tensor | None], dict[str, int]]:
        outputs = {}
        stats = {
            'drugclip_molecule_cache_hit_count': 0,
            'drugclip_molecule_cache_miss_count': 0,
            'drugclip_unique_smiles_success_count': 0,
            'drugclip_unique_smiles_failure_count': 0,
            'drugclip_fail_smiles_parse_count': 0,
            'drugclip_fail_embed_exception_count': 0,
            'drugclip_fail_zero_conformer_count': 0,
            'drugclip_fail_empty_atom_list_count': 0,
        }
        pending_smiles = []
        pending_features = []
        for smiles in smiles_list:
            cached = self._get_from_cache(self._mol_embedding_cache, smiles)
            if cached is not None:
                stats['drugclip_molecule_cache_hit_count'] += 1
                stats['drugclip_unique_smiles_success_count'] += 1
                outputs[smiles] = cached
                continue
            stats['drugclip_molecule_cache_miss_count'] += 1
            feature_result = _prepare_molecule_features(
                smiles,
                num_conformers=self.config.num_conformers,
                conformer_num_workers=self.config.conformer_num_workers,
                seed=self.config.seed + _stable_uint32(smiles),
                dictionary=self.mol_dictionary,
            )
            if feature_result.feature is None:
                failure_reason = feature_result.failure_reason
                if failure_reason is None:
                    raise ValueError('DrugCLIP molecule feature preparation failed without a failure reason')
                stats['drugclip_unique_smiles_failure_count'] += 1
                stats[f'drugclip_fail_{failure_reason}_count'] += 1
                outputs[smiles] = None
                continue
            pending_smiles.append(smiles)
            pending_features.append(feature_result.feature)

        for start in range(0, len(pending_features), self.config.batch_size):
            batch_smiles = pending_smiles[start:start + self.config.batch_size]
            batch_features = pending_features[start:start + self.config.batch_size]
            mol_src_tokens, mol_src_distance, mol_src_edge_type = _collate_features(
                batch_features,
                self.mol_dictionary.pad(),
                self.device,
            )
            with torch.no_grad():
                embeddings = self.model.encode_molecules(
                    mol_src_tokens=mol_src_tokens,
                    mol_src_distance=mol_src_distance,
                    mol_src_edge_type=mol_src_edge_type,
                )
            for smiles, embedding in zip(batch_smiles, embeddings):
                self._put_in_cache(
                    self._mol_embedding_cache,
                    smiles,
                    embedding,
                    self.config.mol_cache_size,
                )
                outputs[smiles] = embedding
                stats['drugclip_unique_smiles_success_count'] += 1
        return outputs, stats

    def _pocket_cache_key(self, pocket_entry: dict) -> int:
        if pocket_entry.get('lmdb_key') is not None:
            return int(pocket_entry['lmdb_key'])
        return int(pocket_entry['source_index'])

    def _encode_missing_pockets(self, pocket_entries: list[dict]) -> dict[int, torch.Tensor]:
        outputs = {}
        pending_keys = []
        pending_features = []
        for pocket_entry in pocket_entries:
            cache_key = self._pocket_cache_key(pocket_entry)
            cached = self._get_from_cache(self._pocket_embedding_cache, cache_key)
            if cached is not None:
                outputs[cache_key] = cached
                continue
            raw_entry = self.raw_entry_store.get_entry(pocket_entry)
            feature = _prepare_pocket_features(
                raw_entry,
                max_pocket_atoms=self.config.max_pocket_atoms,
                seed=self.config.seed,
                source_index=int(pocket_entry['source_index']),
                dictionary=self.pocket_dictionary,
            )
            pending_keys.append(cache_key)
            pending_features.append(feature)

        for start in range(0, len(pending_features), self.config.batch_size):
            batch_keys = pending_keys[start:start + self.config.batch_size]
            batch_features = pending_features[start:start + self.config.batch_size]
            pocket_src_tokens, pocket_src_distance, pocket_src_edge_type = _collate_features(
                batch_features,
                self.pocket_dictionary.pad(),
                self.device,
            )
            with torch.no_grad():
                embeddings = self.model.encode_pockets(
                    pocket_src_tokens=pocket_src_tokens,
                    pocket_src_distance=pocket_src_distance,
                    pocket_src_edge_type=pocket_src_edge_type,
                )
            for cache_key, embedding in zip(batch_keys, embeddings):
                self._put_in_cache(
                    self._pocket_embedding_cache,
                    cache_key,
                    embedding,
                    self.config.pocket_cache_size,
                )
                outputs[cache_key] = embedding
        return outputs

    def score(self, smiles_list: list[str], pocket_entries: list[dict]) -> list[float | None]:
        if len(smiles_list) != len(pocket_entries):
            raise ValueError(
                'DrugCLIP scoring requires smiles_list and pocket_entries to have matching lengths: '
                f'{len(smiles_list)} vs {len(pocket_entries)}'
            )
        if not smiles_list:
            self.last_score_stats = self._empty_last_score_stats()
            return []

        self.last_score_stats = self._empty_last_score_stats()
        self.last_score_stats['drugclip_input_count'] = int(len(smiles_list))
        unique_smiles = list(dict.fromkeys(smiles_list))
        self.last_score_stats['drugclip_unique_smiles_count'] = int(len(unique_smiles))
        molecule_embeddings, molecule_stats = self._encode_missing_molecules(unique_smiles)
        self.last_score_stats.update(molecule_stats)

        unique_pocket_entries = []
        seen_pocket_keys = set()
        for pocket_entry in pocket_entries:
            pocket_key = self._pocket_cache_key(pocket_entry)
            if pocket_key in seen_pocket_keys:
                continue
            seen_pocket_keys.add(pocket_key)
            unique_pocket_entries.append(pocket_entry)
        self.last_score_stats['drugclip_unique_pocket_count'] = int(len(unique_pocket_entries))
        pocket_embeddings = self._encode_missing_pockets(unique_pocket_entries)

        outputs: list[float | None] = []
        for smiles, pocket_entry in zip(smiles_list, pocket_entries):
            mol_embedding = molecule_embeddings.get(smiles)
            if mol_embedding is None:
                self.last_score_stats['drugclip_score_failure_count'] += 1
                outputs.append(None)
                continue
            pocket_embedding = pocket_embeddings[self._pocket_cache_key(pocket_entry)]
            score = torch.sum(mol_embedding.float() * pocket_embedding.float()).item()
            self.last_score_stats['drugclip_score_success_count'] += 1
            outputs.append(float(score))
        if self.last_score_stats['drugclip_input_count'] > 0:
            self.last_score_stats['drugclip_score_success_fraction'] = (
                self.last_score_stats['drugclip_score_success_count'] / self.last_score_stats['drugclip_input_count']
            )
        return outputs
