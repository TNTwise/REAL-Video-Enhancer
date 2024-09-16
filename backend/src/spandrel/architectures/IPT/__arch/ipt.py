# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from spandrel.util import store_hyperparameters

from .common import MeanShift, ResBlock, Upsampler, default_conv


@store_hyperparameters()
class IPT(nn.Module):
    hyperparameters = {}

    def __init__(
        self,
        *,
        patch_size: int = 48,
        patch_dim: int = 3,
        n_feats: int = 64,
        rgb_range: float = 255,
        n_colors: int = 3,
        scale: Sequence[int] = [1],
        num_heads: int = 12,
        num_layers: int = 12,
        num_queries: int = 1,
        dropout_rate: float = 0,
        mlp=True,
        pos_every=False,
        no_pos=False,
        no_norm=False,
        conv=default_conv,
    ):
        super().__init__()

        self.scale_idx = 0
        self.scale = scale
        self.rgb_range = rgb_range
        self.patch_size = patch_size

        kernel_size = 3
        act = nn.ReLU(True)

        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        self.head = nn.ModuleList(
            [
                nn.Sequential(
                    conv(n_colors, n_feats, kernel_size),
                    ResBlock(conv, n_feats, 5, act=act),
                    ResBlock(conv, n_feats, 5, act=act),
                )
                for _ in scale
            ]
        )

        self.body = VisionTransformer(
            img_dim=patch_size,
            patch_dim=patch_dim,
            num_channels=n_feats,
            embedding_dim=n_feats * patch_dim * patch_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=n_feats * patch_dim * patch_dim * 4,
            num_queries=num_queries,
            dropout_rate=dropout_rate,
            mlp=mlp,
            pos_every=pos_every,
            no_pos=no_pos,
            no_norm=no_norm,
        )

        self.tail = nn.ModuleList(
            [
                nn.Sequential(
                    Upsampler(conv, s, n_feats, act=None),
                    conv(n_feats, n_colors, kernel_size),
                )
                for s in scale
            ]
        )

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head[self.scale_idx](x)

        res = self.body(x, self.scale_idx)
        res += x

        x = self.tail[self.scale_idx](res)
        x = self.add_mean(x)

        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_dim: int,
        patch_dim: int,
        num_channels: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        hidden_dim: int,
        num_queries: int,
        dropout_rate: float = 0,
        no_norm=False,
        mlp=True,
        pos_every=False,
        no_pos=False,
    ):
        super().__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0
        self.no_norm = no_norm
        self.mlp = mlp
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels

        self.img_dim = img_dim
        self.pos_every = pos_every
        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * num_channels

        self.out_dim = patch_dim * patch_dim * num_channels

        self.no_pos = no_pos

        if self.mlp:
            self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.out_dim),
                nn.Dropout(dropout_rate),
            )

            self.query_embed = nn.Embedding(
                num_queries, embedding_dim * self.seq_length
            )

        encoder_layer = TransformerEncoderLayer(
            embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = TransformerDecoderLayer(
            embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers)

        if not self.no_pos:
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )

        self.dropout_layer1 = nn.Dropout(dropout_rate)

        if no_norm:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=1 / m.weight.size(1))

    def forward(self, x, query_idx, con=False):
        x = (
            torch.nn.functional.unfold(x, self.patch_dim, stride=self.patch_dim)
            .transpose(1, 2)
            .transpose(0, 1)
            .contiguous()
        )

        if self.mlp:
            x = self.dropout_layer1(self.linear_encoding(x)) + x

            query_embed = (
                self.query_embed.weight[query_idx]
                .view(-1, 1, self.embedding_dim)
                .repeat(1, x.size(1), 1)
            )
        else:
            query_embed = None

        if self.no_pos:
            x = self.encoder(x)
            x = self.decoder(x, x, query_pos=query_embed)
        else:
            pos = self.position_encoding(x).transpose(0, 1)

            if self.pos_every:
                x = self.encoder(x, pos=pos)
                x = self.decoder(x, x, pos=pos, query_pos=query_embed)
            else:
                x = self.encoder(x + pos)
                x = self.decoder(x, x, query_pos=query_embed)

        if self.mlp:
            x = self.mlp_head(x) + x

        x = x.transpose(0, 1).contiguous().view(x.size(1), -1, self.flatten_dim)

        if con:
            con_x = x
            x = torch.nn.functional.fold(
                x.transpose(1, 2).contiguous(),
                int(self.img_dim),
                self.patch_dim,
                stride=self.patch_dim,
            )
            return x, con_x

        x = torch.nn.functional.fold(
            x.transpose(1, 2).contiguous(),
            int(self.img_dim),
            self.patch_dim,
            stride=self.patch_dim,
        )

        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(
        self, max_position_embeddings: int, embedding_dim: int, seq_length: int
    ):
        super().__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids", torch.arange(self.seq_length).expand((1, -1))
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)
        return position_embeddings


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, pos=None):
        output = src

        for layer in self.layers:
            output = layer(output, pos=pos)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward=2048,
        dropout=0.1,
        no_norm=False,
        activation="relu",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, bias=False
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        nn.init.kaiming_uniform_(self.self_attn.in_proj_weight, a=math.sqrt(5))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, src2)
        src = src + self.dropout1(src2[0])
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, pos=None, query_pos=None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead: int,
        dim_feedforward=2048,
        dropout=0.1,
        no_norm=False,
        activation="relu",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, bias=False
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, bias=False
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos=None, query_pos=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


def _get_clones(module: nn.Module, N: int):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation: str):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
