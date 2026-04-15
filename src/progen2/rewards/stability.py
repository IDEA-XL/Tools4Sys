from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import BertTokenizer

from progen2.rewards.common import iter_chunks, release_model, validate_batch_size


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


class _TemBERTureReplica:
    def __init__(self, replica_dir, *, device, batch_size, base_model_name):
        self.device = torch.device(device)
        self.batch_size = validate_batch_size(batch_size, field_name='stability.batch_size')
        self.replica_dir = Path(replica_dir).resolve()
        self.base_model_name = str(base_model_name)
        if not self.replica_dir.is_dir():
            raise ValueError(f'TemBERTure replica directory not found: {self.replica_dir}')
        adapter_dir = self.replica_dir / 'AdapterBERT_adapter'
        head_dir = self.replica_dir / 'AdapterBERT_head_adapter'
        if not adapter_dir.is_dir():
            raise ValueError(f'TemBERTure adapter directory not found: {adapter_dir}')
        if not head_dir.is_dir():
            raise ValueError(f'TemBERTure head directory not found: {head_dir}')

        BertAdapterModel = _import_adapter_model()
        self.model = BertAdapterModel.from_pretrained(self.base_model_name)
        self.model.load_adapter(str(adapter_dir), with_head=True)
        self.model.load_head(str(head_dir))
        self.model.set_active_adapters(['AdapterBERT_adapter'])
        self.model.train_adapter(['AdapterBERT_adapter'])
        if hasattr(self.model, 'heads') and 'default' in self.model.heads:
            self.model.delete_head('default')
        if hasattr(self.model, 'bert') and hasattr(self.model.bert, 'prompt_tuning'):
            self.model.bert.prompt_tuning = nn.Identity()
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(self.base_model_name)

    @torch.no_grad()
    def score_raw(self, sequences):
        if not sequences:
            raise ValueError('sequences must be non-empty')
        self.model.to(self.device)
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
        release_model(self.model, self.device)


class TemBERTureTmScorer:
    def __init__(
        self,
        model_name_or_path,
        tokenizer_name_or_path=None,
        device='cpu',
        batch_size=16,
        base_model_name='Rostlab/prot_bert_bfd',
        replicas=None,
    ):
        if tokenizer_name_or_path is not None:
            raise ValueError('TemBERTure uses the official ProtBERT tokenizer; tokenizer_name_or_path must be omitted')
        if not model_name_or_path:
            raise ValueError('TemBERTure model_name_or_path is required')
        self.device = torch.device(device)
        self.batch_size = validate_batch_size(batch_size, field_name='stability.batch_size')
        self.root_dir = _resolve_temberture_root(model_name_or_path)
        self.base_model_name = str(base_model_name)
        self.replica_names = list(replicas or ['replica1', 'replica2', 'replica3'])
        if not self.replica_names:
            raise ValueError('TemBERTure replicas must be non-empty')
        self.replicas = None

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
                    base_model_name=self.base_model_name,
                )
            )
        self.replicas = loaded

    def release(self):
        if self.replicas is None:
            return
        for replica in self.replicas:
            replica.release()
        self.replicas = None

    @torch.no_grad()
    def score_raw(self, sequences):
        if not sequences:
            raise ValueError('sequences must be non-empty')
        self._ensure_loaded()
        replica_scores = [replica.score_raw(sequences) for replica in self.replicas]
        if not replica_scores:
            raise RuntimeError('TemBERTure scorer loaded no replicas')
        tensor = torch.tensor(replica_scores, dtype=torch.float32)
        return tensor.mean(dim=0).tolist()
