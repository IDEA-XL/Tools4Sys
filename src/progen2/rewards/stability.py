from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import BertTokenizer

from progen2.rewards.common import iter_chunks, move_model_to_device, release_model, validate_batch_size


def _import_adapter_model():
    try:
        from adapters import BertAdapterModel
    except ImportError as exc:
        raise RuntimeError(
            "TemBERTure scoring requires the 'adapters' package. "
            'Install it in the runtime environment before running ProGen2 SGRPO.'
        ) from exc
    return BertAdapterModel


def _resolve_temberture_root(model_name_or_path):
    root = Path(model_name_or_path).expanduser().resolve()
    candidates = [root, root / 'temBERTure']
    for candidate in candidates:
        if (candidate / 'temBERTure_TM' / 'replica1').is_dir():
            return candidate
    raise ValueError(
        'TemBERTure model_name_or_path must point to the official TemBERTure checkout root '
        f'or its temBERTure subdirectory; missing temBERTure_TM/replica1 under {root}'
    )


def _resolve_local_protbert_root(model_name_or_path):
    if not model_name_or_path:
        raise ValueError(
            'TemBERTure requires an explicit local ProtBERT directory via '
            'stability.base_model_name_or_path; remote Hugging Face model IDs are not allowed'
        )
    root = Path(model_name_or_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f'ProtBERT directory not found: {root}')
    required_files = [
        'config.json',
        'pytorch_model.bin',
        'vocab.txt',
        'tokenizer_config.json',
        'special_tokens_map.json',
    ]
    missing = [name for name in required_files if not (root / name).is_file()]
    if missing:
        raise ValueError(
            'ProtBERT directory is incomplete; expected files missing under '
            f'{root}: {missing}'
        )
    return root


class _TemBERTureReplica:
    def __init__(self, replica_dir, *, device, batch_size, base_model_path):
        self.device = torch.device(device)
        self.batch_size = validate_batch_size(batch_size, field_name='stability.batch_size')
        self.replica_dir = Path(replica_dir).resolve()
        self.base_model_path = str(base_model_path)
        if not self.replica_dir.is_dir():
            raise ValueError(f'TemBERTure replica directory not found: {self.replica_dir}')
        adapter_dir = self.replica_dir / 'AdapterBERT_adapter'
        head_dir = self.replica_dir / 'AdapterBERT_head_adapter'
        if not adapter_dir.is_dir():
            raise ValueError(f'TemBERTure adapter directory not found: {adapter_dir}')
        if not head_dir.is_dir():
            raise ValueError(f'TemBERTure head directory not found: {head_dir}')

        BertAdapterModel = _import_adapter_model()
        self.model = BertAdapterModel.from_pretrained(self.base_model_path)
        self.model.load_adapter(str(adapter_dir), with_head=True)
        self.model.load_head(str(head_dir))
        self.model.set_active_adapters(['AdapterBERT_adapter'])
        self.model.train_adapter(['AdapterBERT_adapter'])
        if hasattr(self.model, 'heads') and 'default' in self.model.heads:
            self.model.delete_head('default')
        if hasattr(self.model, 'bert') and hasattr(self.model.bert, 'prompt_tuning'):
            self.model.bert.prompt_tuning = nn.Identity()
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(self.base_model_path)
        self.last_move_to_device_sec = 0.0
        self.last_release_to_cpu_sec = 0.0

    @torch.no_grad()
    def score_raw(self, sequences):
        if not sequences:
            raise ValueError('sequences must be non-empty')
        self.last_move_to_device_sec = move_model_to_device(self.model, self.device)
        outputs = []
        normalized = [' '.join(''.join(sequence.split())) for sequence in sequences]
        for chunk in iter_chunks(normalized, self.batch_size):
            batch = self.tokenizer(
                chunk,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512,
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            logits = self.model(**batch).logits
            outputs.extend(logits.reshape(-1).detach().cpu().tolist())
        return outputs

    def release(self):
        self.last_release_to_cpu_sec = release_model(self.model, self.device)


class TemBERTureTmScorer:
    def __init__(
        self,
        model_name_or_path,
        tokenizer_name_or_path=None,
        device='cpu',
        batch_size=16,
        base_model_name_or_path=None,
        base_model_name=None,
        replicas=None,
    ):
        if tokenizer_name_or_path is not None:
            raise ValueError('TemBERTure uses the official ProtBERT tokenizer; tokenizer_name_or_path must be omitted')
        if not model_name_or_path:
            raise ValueError('TemBERTure model_name_or_path is required')
        self.device = torch.device(device)
        self.batch_size = validate_batch_size(batch_size, field_name='stability.batch_size')
        self.root_dir = _resolve_temberture_root(model_name_or_path)
        base_model_root = base_model_name_or_path if base_model_name_or_path is not None else base_model_name
        self.base_model_path = _resolve_local_protbert_root(base_model_root)
        self.replica_names = list(replicas or ['replica1', 'replica2', 'replica3'])
        if not self.replica_names:
            raise ValueError('TemBERTure replicas must be non-empty')
        self.replicas = None
        self.last_move_to_device_sec = 0.0
        self.last_release_to_cpu_sec = 0.0

    def _ensure_loaded(self):
        if self.replicas is not None:
            return
        loaded = []
        for replica_name in self.replica_names:
            replica_dir = self.root_dir / 'temBERTure_TM' / replica_name
            loaded.append(
                _TemBERTureReplica(
                    replica_dir,
                    device=self.device,
                    batch_size=self.batch_size,
                    base_model_path=self.base_model_path,
                )
            )
        self.replicas = loaded

    def release(self):
        if self.replicas is None:
            return
        self.last_release_to_cpu_sec = 0.0
        for replica in self.replicas:
            replica.release()
            self.last_release_to_cpu_sec += replica.last_release_to_cpu_sec

    @torch.no_grad()
    def score_raw(self, sequences):
        if not sequences:
            raise ValueError('sequences must be non-empty')
        self._ensure_loaded()
        self.last_move_to_device_sec = 0.0
        replica_scores = [replica.score_raw(sequences) for replica in self.replicas]
        for replica in self.replicas:
            self.last_move_to_device_sec += replica.last_move_to_device_sec
        if not replica_scores:
            raise RuntimeError('TemBERTure scorer loaded no replicas')
        tensor = torch.tensor(replica_scores, dtype=torch.float32)
        return tensor.mean(dim=0).tolist()
