from __future__ import annotations

import math
from typing import Sequence

import torch
from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.ipt import IPT
from .__arch.model import Model


class IPTArch(Architecture[IPT]):
    def __init__(self) -> None:
        super().__init__(
            id="IPT",
            detect=KeyCondition.has_all(
                "sub_mean.weight",
                "sub_mean.bias",
                "add_mean.weight",
                "add_mean.bias",
                "head.0.0.weight",
                "head.0.0.bias",
                "head.0.1.body.0.weight",
                "head.0.1.body.0.bias",
                "head.0.1.body.2.weight",
                "head.0.1.body.2.bias",
                "head.0.2.body.0.weight",
                "head.0.2.body.2.weight",
                "body.encoder.layers.0.self_attn.in_proj_weight",
                "body.encoder.layers.0.self_attn.out_proj.weight",
                "body.encoder.layers.0.linear1.weight",
                "body.encoder.layers.0.linear2.weight",
                "body.decoder.layers.0.self_attn.in_proj_weight",
                "body.decoder.layers.0.self_attn.out_proj.weight",
                "body.decoder.layers.0.multihead_attn.in_proj_weight",
                "body.decoder.layers.0.multihead_attn.out_proj.weight",
                "body.decoder.layers.0.linear1.weight",
                "body.decoder.layers.0.linear2.weight",
                "tail.0.1.weight",
                "tail.0.1.bias",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[IPT]:
        # patch_size: int
        # patch_dim: int
        # n_feats: int
        # rgb_range: float
        scale: Sequence[int] = [1]
        num_heads: int = 12
        # num_layers: int
        # num_queries: int
        # mlp = True
        pos_every = False  # cannot be deduced from state_dict
        # no_pos = False
        # no_norm = False

        n_feats = state_dict["head.0.0.weight"].shape[0]
        # embedding_dim = n_feats * patch_dim * patch_dim
        embedding_dim = state_dict[
            "body.encoder.layers.0.self_attn.in_proj_weight"
        ].shape[1]
        patch_dim = math.isqrt(embedding_dim // n_feats)

        num_layers = get_seq_len(state_dict, "body.encoder.layers")
        mlp = "body.linear_encoding.weight" in state_dict
        no_pos = "body.position_encoding.position_ids" not in state_dict
        no_norm = "body.encoder.layers.0.norm1.weight" not in state_dict

        if mlp:
            num_queries = state_dict["body.query_embed.weight"].shape[0]
            seq_length = state_dict["body.query_embed.weight"].shape[1] // embedding_dim
            patch_size = math.isqrt(seq_length) * patch_dim
        else:
            # those numbers cannot be deduced without mlp, so just set them to their defaults
            num_queries = 1
            patch_size = 48

        if not no_pos:
            seq_length = state_dict["body.position_encoding.position_ids"].shape[1]
            patch_size = math.isqrt(seq_length) * patch_dim

        # read bias to deduce rgb_range
        # bias = -rgb_range * torch.Tensor((0.4488, 0.4371, 0.4040))
        bias: torch.Tensor = state_dict["sub_mean.bias"]
        rgb_range_t = -bias / torch.Tensor((0.4488, 0.4371, 0.4040))
        rgb_range_n = rgb_range_t.mean().detach().item()
        rgb_range = round(rgb_range_n)

        scale_count = get_seq_len(state_dict, "tail")
        assert scale_count > 0
        scale = []
        for i in range(scale_count):
            s = 1
            for j in range(5):
                key = f"tail.{i}.0.{j*2}.weight"
                if key in state_dict:
                    shape = state_dict[key].shape
                    s *= math.isqrt(shape[0] // shape[1])
            scale.append(s)

        model = IPT(
            patch_size=patch_size,
            patch_dim=patch_dim,
            n_feats=n_feats,
            rgb_range=rgb_range,
            n_colors=3,  # must be RGB
            scale=scale,
            num_heads=num_heads,
            num_layers=num_layers,
            num_queries=num_queries,
            mlp=mlp,
            pos_every=pos_every,
            no_pos=no_pos,
            no_norm=no_norm,
        )

        single_scale = max(scale)

        def call(model: IPT, x: torch.Tensor):
            m = Model(model)

            scale_idx = model.scale.index(max(model.scale))
            return m.forward(x * model.rgb_range, scale_idx) / model.rgb_range

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if single_scale == 1 else "SR",
            tags=[
                f"{n_feats}nf",
                f"{patch_dim}pd",
                f"{num_heads}nh",
                f"{num_layers}nl",
            ],
            supports_half=False,  # TODO: verify
            supports_bfloat16=True,
            scale=single_scale,
            input_channels=3,  # only supports RGB
            output_channels=3,
            size_requirements=SizeRequirements(minimum=patch_size),
            call_fn=call,
        )


__all__ = ["IPTArch", "IPT"]
