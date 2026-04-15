import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TemBERTureTmScorer:
    def __init__(self, model_name_or_path, tokenizer_name_or_path=None, device='cpu'):
        if not model_name_or_path:
            raise ValueError('TemBERTure model_name_or_path is required')
        self.device = torch.device(device)
        tokenizer_name_or_path = tokenizer_name_or_path or model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.model.eval().to(self.device)

    @torch.no_grad()
    def score_raw(self, sequences):
        if not sequences:
            raise ValueError('sequences must be non-empty')
        batch = self.tokenizer(
            sequences,
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
        return logits[:, 0].detach().cpu().tolist()
