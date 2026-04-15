import torch


class ESM2NaturalnessScorer:
    def __init__(self, model_name='esm2_t33_650M_UR50D', device='cpu'):
        try:
            import esm
        except ImportError as exc:
            raise ImportError('esm is required for ESM2 naturalness scoring') from exc

        self.device = torch.device(device)
        self.model_name = str(model_name)
        if not hasattr(esm.pretrained, self.model_name):
            raise ValueError(f'Unsupported ESM2 model name: {self.model_name}')
        loader = getattr(esm.pretrained, self.model_name)
        self.model, self.alphabet = loader()
        self.model.eval().to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()

    @torch.no_grad()
    def score_raw(self, sequences):
        if not sequences:
            raise ValueError('sequences must be non-empty')
        batch = [(str(index), sequence) for index, sequence in enumerate(sequences)]
        _, _, tokens = self.batch_converter(batch)
        tokens = tokens.to(self.device)
        outputs = self.model(tokens, repr_layers=[], return_contacts=False)
        logits = outputs['logits']
        log_probs = torch.log_softmax(logits, dim=-1)
        scores = []
        for row_idx, sequence in enumerate(sequences):
            token_ids = tokens[row_idx, 1:1 + len(sequence)]
            token_log_probs = log_probs[row_idx, 1:1 + len(sequence)]
            gathered = token_log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
            scores.append(gathered.mean().item())
        return scores
