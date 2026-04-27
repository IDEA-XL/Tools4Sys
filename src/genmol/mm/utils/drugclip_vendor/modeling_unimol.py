"""Vendored and adapted from the official DrugCLIP repository.

Source of truth:
- https://github.com/bowen-gao/DrugCLIP
- unimol/models/unimol.py
- unimol/models/transformer_encoder_with_pair.py
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore.modules import TransformerEncoderLayer


def _get_activation_fn(name: str):
    if name == 'relu':
        return F.relu
    if name == 'gelu':
        return F.gelu
    if name == 'tanh':
        return torch.tanh
    if name == 'silu':
        return F.silu
    raise ValueError(f'Unsupported DrugCLIP activation function: {name!r}')


class TransformerEncoderWithPair(nn.Module):
    def __init__(
        self,
        *,
        encoder_layers: int = 6,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        emb_dropout: float = 0.1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        max_seq_len: int = 256,
        activation_fn: str = 'gelu',
        post_ln: bool = False,
        no_final_head_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.emb_dropout = float(emb_dropout)
        self.max_seq_len = int(max_seq_len)
        self.embed_dim = int(embed_dim)
        self.attention_heads = int(attention_heads)
        self.emb_layer_norm = nn.LayerNorm(self.embed_dim)
        self.final_layer_norm = None if post_ln else nn.LayerNorm(self.embed_dim)
        self.final_head_layer_norm = None if no_final_head_layer_norm else nn.LayerNorm(attention_heads)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=ffn_embed_dim,
                    attention_heads=attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    post_ln=post_ln,
                )
                for _ in range(encoder_layers)
            ]
        )

    def forward(
        self,
        emb: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        batch_size = emb.size(0)
        seq_len = emb.size(1)
        x = self.emb_layer_norm(emb)
        x = F.dropout(x, p=self.emb_dropout, training=self.training)
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        input_attn_mask = attn_mask
        input_padding_mask = padding_mask

        def fill_attn_mask(current_attn_mask, current_padding_mask, fill_value=float('-inf')):
            if current_attn_mask is not None and current_padding_mask is not None:
                current_attn_mask = current_attn_mask.view(x.size(0), -1, seq_len, seq_len)
                current_attn_mask.masked_fill_(
                    current_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    fill_value,
                )
                current_attn_mask = current_attn_mask.view(-1, seq_len, seq_len)
                current_padding_mask = None
            return current_attn_mask, current_padding_mask

        if attn_mask is None:
            raise ValueError('DrugCLIP encoder requires an attention bias tensor')
        attn_mask, padding_mask = fill_attn_mask(attn_mask, padding_mask)

        for layer in self.layers:
            x, attn_mask, _ = layer(
                x,
                padding_mask=padding_mask,
                attn_bias=attn_mask,
                return_attn=True,
            )

        def norm_loss(tensor, eps=1e-10, tolerance=1.0):
            tensor = tensor.float()
            max_norm = tensor.shape[-1] ** 0.5
            norm = torch.sqrt(torch.sum(tensor**2, dim=-1) + eps)
            error = torch.nn.functional.relu((norm - max_norm).abs() - tolerance)
            return error

        def masked_mean(mask, value, dim=-1, eps=1e-10):
            return (torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))).mean()

        x_norm = norm_loss(x)
        if input_padding_mask is not None:
            token_mask = 1.0 - input_padding_mask.float()
        else:
            token_mask = torch.ones_like(x_norm, device=x.device)
        x_norm = masked_mean(token_mask, x_norm)

        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        delta_pair_repr = attn_mask - input_attn_mask
        delta_pair_repr, _ = fill_attn_mask(delta_pair_repr, input_padding_mask, 0.0)
        attn_mask = attn_mask.view(batch_size, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        delta_pair_repr = (
            delta_pair_repr.view(batch_size, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        )

        pair_mask = token_mask[..., None] * token_mask[..., None, :]
        delta_pair_repr_norm = norm_loss(delta_pair_repr)
        delta_pair_repr_norm = masked_mean(pair_mask, delta_pair_repr_norm, dim=(-1, -2))

        if self.final_head_layer_norm is not None:
            delta_pair_repr = self.final_head_layer_norm(delta_pair_repr)

        return x, attn_mask, delta_pair_repr, x_norm, delta_pair_repr_norm


class NonLinearHead(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, activation_fn: str, hidden: int | None = None):
        super().__init__()
        hidden_dim = input_dim if hidden is None else int(hidden)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.activation_fn = _get_activation_fn(activation_fn)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.linear1(inputs)
        outputs = self.activation_fn(outputs)
        outputs = self.linear2(outputs)
        return outputs


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    normalizer = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (normalizer * std)


class GaussianLayer(nn.Module):
    def __init__(self, num_bases: int = 128, edge_types: int = 1024):
        super().__init__()
        self.num_bases = int(num_bases)
        self.means = nn.Embedding(1, self.num_bases)
        self.stds = nn.Embedding(1, self.num_bases)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, distances: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        mul = self.mul(edge_type).type_as(distances)
        bias = self.bias(edge_type).type_as(distances)
        values = mul * distances.unsqueeze(-1) + bias
        values = values.expand(-1, -1, -1, self.num_bases)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(values.float(), mean, std).type_as(self.means.weight)


class UniMolEncoder(nn.Module):
    def __init__(self, config, *, vocab_size: int, padding_idx: int):
        super().__init__()
        self.padding_idx = int(padding_idx)
        self.embed_tokens = nn.Embedding(vocab_size, int(config.encoder_embed_dim), self.padding_idx)
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=int(config.encoder_layers),
            embed_dim=int(config.encoder_embed_dim),
            ffn_embed_dim=int(config.encoder_ffn_embed_dim),
            attention_heads=int(config.encoder_attention_heads),
            emb_dropout=float(config.emb_dropout),
            dropout=float(config.dropout),
            attention_dropout=float(config.attention_dropout),
            activation_dropout=float(config.activation_dropout),
            max_seq_len=int(config.max_seq_len),
            activation_fn=str(config.activation_fn),
            post_ln=bool(config.post_ln),
            no_final_head_layer_norm=float(config.delta_pair_repr_norm_loss) < 0.0,
        )
        self.gbf_proj = NonLinearHead(128, int(config.encoder_attention_heads), str(config.activation_fn))
        self.gbf = GaussianLayer(128, vocab_size * vocab_size)

    def encode(
        self,
        *,
        src_tokens: torch.Tensor,
        src_distance: torch.Tensor,
        src_edge_type: torch.Tensor,
    ) -> torch.Tensor:
        padding_mask = src_tokens.eq(self.padding_idx)
        embeddings = self.embed_tokens(src_tokens)
        num_nodes = src_distance.size(-1)
        gbf_feature = self.gbf(src_distance, src_edge_type)
        gbf_result = self.gbf_proj(gbf_feature)
        graph_attn_bias = gbf_result.permute(0, 3, 1, 2).contiguous().view(-1, num_nodes, num_nodes)
        encoder_rep, _, _, _, _ = self.encoder(
            embeddings,
            padding_mask=padding_mask,
            attn_mask=graph_attn_bias,
        )
        return encoder_rep
