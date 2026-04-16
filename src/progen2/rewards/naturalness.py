import logging
from pathlib import Path

import torch

from progen2.rewards.common import iter_chunks, release_model, validate_batch_size


logger = logging.getLogger(__name__)


class ESM2NaturalnessScorer:
    def __init__(self, model_name='esm2_t33_650M_UR50D', device='cpu', batch_size=8):
        try:
            import esm
        except ImportError as exc:
            raise ImportError('esm is required for ESM2 naturalness scoring') from exc

        self.esm = esm
        self.device = torch.device(device)
        self.model_name = str(model_name)
        self.batch_size = validate_batch_size(batch_size, field_name='naturalness.batch_size')
        if not hasattr(esm.pretrained, self.model_name):
            raise ValueError(f'Unsupported ESM2 model name: {self.model_name}')
        self.model = None
        self.alphabet = None
        self.batch_converter = None

    def _cached_checkpoint_path(self):
        return Path(torch.hub.get_dir()) / 'checkpoints' / f'{self.model_name}.pt'

    def _load_model(self):
        loader = getattr(self.esm.pretrained, self.model_name)
        return loader()

    def _ensure_loaded(self):
        if self.model is None:
            try:
                self.model, self.alphabet = self._load_model()
            except OSError as exc:
                cache_path = self._cached_checkpoint_path()
                if not cache_path.exists():
                    raise
                logger.warning(
                    'Removing corrupted ESM2 checkpoint cache and retrying once: %s (%s)',
                    cache_path,
                    exc,
                )
                cache_path.unlink()
                self.model, self.alphabet = self._load_model()
            self.model.eval()
            self.batch_converter = self.alphabet.get_batch_converter()
        self.model.to(self.device)

    def release(self):
        release_model(self.model, self.device)

    @torch.no_grad()
    def score_raw(self, sequences):
        if not sequences:
            raise ValueError('sequences must be non-empty')
        scores = []
        self._ensure_loaded()
        for start_idx, chunk in enumerate(iter_chunks(sequences, self.batch_size)):
            batch = [
                (str(start_idx * self.batch_size + row_idx), sequence)
                for row_idx, sequence in enumerate(chunk)
            ]
            _, _, tokens = self.batch_converter(batch)
            tokens = tokens.to(self.device)
            outputs = self.model(tokens, repr_layers=[], return_contacts=False)
            logits = outputs['logits']
            log_probs = torch.log_softmax(logits, dim=-1)
            for row_idx, sequence in enumerate(chunk):
                token_ids = tokens[row_idx, 1:1 + len(sequence)]
                token_log_probs = log_probs[row_idx, 1:1 + len(sequence)]
                gathered = token_log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
                scores.append(gathered.mean().item())
        return scores
