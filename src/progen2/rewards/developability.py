import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from progen2.rewards.liability import liability_reward


class ProteinSolScorer:
    def __init__(self, model_name_or_path, tokenizer_name_or_path=None, device='cpu'):
        if not model_name_or_path:
            raise ValueError('Protein-Sol model_name_or_path is required')
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
                'Protein-Sol scorer expects a single scalar score per sequence, '
                f'got shape {list(logits.shape)}'
            )
        return logits[:, 0].detach().cpu().tolist()


def developability_reward(proteinsol_scores, sequences):
    if len(proteinsol_scores) != len(sequences):
        raise ValueError('proteinsol_scores length must match sequences length')
    outputs = []
    for raw_score, sequence in zip(proteinsol_scores, sequences):
        sol = max(0.0, min(1.0, float(raw_score)))
        liab = float(liability_reward(sequence))
        outputs.append(0.8 * sol + 0.2 * liab)
    return outputs
