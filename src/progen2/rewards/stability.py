import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from progen2.rewards.common import iter_chunks, release_model, validate_batch_size


class TemBERTureTmScorer:
    def __init__(self, model_name_or_path, tokenizer_name_or_path=None, device='cpu', batch_size=16):
        if not model_name_or_path:
            raise ValueError('TemBERTure model_name_or_path is required')
        self.device = torch.device(device)
        self.batch_size = validate_batch_size(batch_size, field_name='stability.batch_size')
        self.model_name_or_path = str(model_name_or_path)
        tokenizer_name_or_path = tokenizer_name_or_path or model_name_or_path
        self.tokenizer_name_or_path = str(tokenizer_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
        self.model = None

    def _ensure_loaded(self):
        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)
            self.model.eval()
        self.model.to(self.device)

    def release(self):
        release_model(self.model, self.device)

    @torch.no_grad()
    def score_raw(self, sequences):
        if not sequences:
            raise ValueError('sequences must be non-empty')
        self._ensure_loaded()
        outputs = []
        for chunk in iter_chunks(sequences, self.batch_size):
            batch = self.tokenizer(
                chunk,
                return_tensors='pt',
                padding=True,
                truncation=True,
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            logits = self.model(**batch).logits
            if logits.dim() != 2 or logits.size(1) != 1:
                raise ValueError(
                    'TemBERTure scorer expects a single regression logit per sequence, '
                    f'got shape {list(logits.shape)}'
                )
            outputs.extend(logits[:, 0].detach().cpu().tolist())
        return outputs
