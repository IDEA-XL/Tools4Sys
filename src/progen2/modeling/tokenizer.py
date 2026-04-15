from dataclasses import dataclass

import torch
from tokenizers import Tokenizer


@dataclass(frozen=True)
class TokenizerBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class OfficialProGen2Tokenizer:
    def __init__(self, tokenizer_path):
        self.tokenizer_path = str(tokenizer_path)
        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)

        self.pad_token_id = self._required_special_id('<|pad|>')
        self.bos_token_id = self._required_special_id('1')
        self.eos_token_id = self._required_special_id('2')

    def _required_special_id(self, token):
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f'Official ProGen2 tokenizer is missing required token {token!r}')
        return int(token_id)

    def encode(self, text):
        encoded = self.tokenizer.encode(text)
        return list(encoded.ids)

    def decode(self, token_ids):
        return self.tokenizer.decode([int(token_id) for token_id in token_ids])

    def decode_batch(self, token_id_batch):
        return self.tokenizer.decode_batch(
            [[int(token_id) for token_id in token_ids] for token_ids in token_id_batch]
        )

    def batch_encode(self, texts, device=None):
        if not texts:
            raise ValueError('texts must be non-empty')
        encoded = [self.encode(text) for text in texts]
        max_len = max(len(item) for item in encoded)
        input_ids = []
        attention_mask = []
        for item in encoded:
            pad_len = max_len - len(item)
            input_ids.append(item + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(item) + [0] * pad_len)
        return TokenizerBatch(
            input_ids=torch.tensor(input_ids, dtype=torch.long, device=device),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long, device=device),
        )
