"""Vendored and adapted from the official DrugCLIP repository.

Source of truth:
- https://github.com/bowen-gao/DrugCLIP
- unimol/models/drugclip.py
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modeling_unimol import NonLinearHead, UniMolEncoder


def build_default_drugclip_args(*, max_seq_len: int = 512):
    args = SimpleNamespace()
    args.mol = SimpleNamespace()
    args.pocket = SimpleNamespace()

    args.mol.encoder_layers = 15
    args.mol.encoder_embed_dim = 512
    args.mol.encoder_ffn_embed_dim = 2048
    args.mol.encoder_attention_heads = 64
    args.mol.dropout = 0.1
    args.mol.emb_dropout = 0.1
    args.mol.attention_dropout = 0.1
    args.mol.activation_dropout = 0.0
    args.mol.pooler_dropout = 0.0
    args.mol.max_seq_len = int(max_seq_len)
    args.mol.activation_fn = 'gelu'
    args.mol.pooler_activation_fn = 'tanh'
    args.mol.post_ln = False
    args.mol.masked_token_loss = -1.0
    args.mol.masked_coord_loss = -1.0
    args.mol.masked_dist_loss = -1.0
    args.mol.x_norm_loss = -1.0
    args.mol.delta_pair_repr_norm_loss = -1.0

    args.pocket.encoder_layers = 15
    args.pocket.encoder_embed_dim = 512
    args.pocket.encoder_ffn_embed_dim = 2048
    args.pocket.encoder_attention_heads = 64
    args.pocket.dropout = 0.1
    args.pocket.emb_dropout = 0.1
    args.pocket.attention_dropout = 0.1
    args.pocket.activation_dropout = 0.0
    args.pocket.pooler_dropout = 0.0
    args.pocket.max_seq_len = int(max_seq_len)
    args.pocket.activation_fn = 'gelu'
    args.pocket.pooler_activation_fn = 'tanh'
    args.pocket.post_ln = False
    args.pocket.masked_token_loss = -1.0
    args.pocket.masked_coord_loss = -1.0
    args.pocket.masked_dist_loss = -1.0
    args.pocket.x_norm_loss = -1.0
    args.pocket.delta_pair_repr_norm_loss = -1.0

    return args


class DrugCLIPEncoderModel(nn.Module):
    def __init__(
        self,
        args,
        *,
        mol_vocab_size: int,
        pocket_vocab_size: int,
        mol_padding_idx: int,
        pocket_padding_idx: int,
    ):
        super().__init__()
        self.args = args
        self.mol_model = UniMolEncoder(
            args.mol,
            vocab_size=mol_vocab_size,
            padding_idx=mol_padding_idx,
        )
        self.pocket_model = UniMolEncoder(
            args.pocket,
            vocab_size=pocket_vocab_size,
            padding_idx=pocket_padding_idx,
        )
        self.mol_project = NonLinearHead(int(args.mol.encoder_embed_dim), 128, 'relu')
        self.pocket_project = NonLinearHead(int(args.pocket.encoder_embed_dim), 128, 'relu')

    def encode_molecules(
        self,
        *,
        mol_src_tokens: torch.Tensor,
        mol_src_distance: torch.Tensor,
        mol_src_edge_type: torch.Tensor,
    ) -> torch.Tensor:
        mol_encoder_rep = self.mol_model.encode(
            src_tokens=mol_src_tokens,
            src_distance=mol_src_distance,
            src_edge_type=mol_src_edge_type,
        )
        mol_rep = mol_encoder_rep[:, 0, :]
        mol_emb = self.mol_project(mol_rep)
        return F.normalize(mol_emb, dim=-1)

    def encode_pockets(
        self,
        *,
        pocket_src_tokens: torch.Tensor,
        pocket_src_distance: torch.Tensor,
        pocket_src_edge_type: torch.Tensor,
    ) -> torch.Tensor:
        pocket_encoder_rep = self.pocket_model.encode(
            src_tokens=pocket_src_tokens,
            src_distance=pocket_src_distance,
            src_edge_type=pocket_src_edge_type,
        )
        pocket_rep = pocket_encoder_rep[:, 0, :]
        pocket_emb = self.pocket_project(pocket_rep)
        return F.normalize(pocket_emb, dim=-1)
