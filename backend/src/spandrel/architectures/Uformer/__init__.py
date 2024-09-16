import math

from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.Uformer import Uformer


class UformerArch(Architecture[Uformer]):
    def __init__(self) -> None:
        super().__init__(
            id="Uformer",
            detect=KeyCondition.has_all(
                "input_proj.proj.0.weight",
                "output_proj.proj.0.weight",
                "encoderlayer_0.blocks.0.norm1.weight",
                "encoderlayer_2.blocks.0.norm1.weight",
                "conv.blocks.0.norm1.weight",
                "decoderlayer_0.blocks.0.norm1.weight",
                "decoderlayer_2.blocks.0.norm1.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[Uformer]:
        img_size = 256  # cannot be deduced from state_dict
        in_chans = 3
        dd_in = 3
        embed_dim = 32
        depths = [2, 2, 2, 2, 2, 2, 2, 2, 2]
        num_heads = [1, 2, 4, 8, 16, 16, 8, 4, 2]
        win_size = 8
        mlp_ratio = 4.0
        qkv_bias = True
        drop_rate = 0.0  # cannot be deduced from state_dict
        attn_drop_rate = 0.0  # cannot be deduced from state_dict
        drop_path_rate = 0.1  # cannot be deduced from state_dict
        token_projection = "linear"
        token_mlp = "leff"
        shift_flag = True  # cannot be deduced from state_dict
        modulator = False
        cross_modulator = False

        embed_dim = state_dict["input_proj.proj.0.weight"].shape[0]
        dd_in = state_dict["input_proj.proj.0.weight"].shape[1]
        in_chans = state_dict["output_proj.proj.0.weight"].shape[0]

        depths[0] = get_seq_len(state_dict, "encoderlayer_0.blocks")
        depths[1] = get_seq_len(state_dict, "encoderlayer_1.blocks")
        depths[2] = get_seq_len(state_dict, "encoderlayer_2.blocks")
        depths[3] = get_seq_len(state_dict, "encoderlayer_3.blocks")
        depths[4] = get_seq_len(state_dict, "conv.blocks")
        depths[5] = get_seq_len(state_dict, "decoderlayer_0.blocks")
        depths[6] = get_seq_len(state_dict, "decoderlayer_1.blocks")
        depths[7] = get_seq_len(state_dict, "decoderlayer_2.blocks")
        depths[8] = get_seq_len(state_dict, "decoderlayer_3.blocks")

        num_heads_suffix = "blocks.0.attn.relative_position_bias_table"
        num_heads[0] = state_dict[f"encoderlayer_0.{num_heads_suffix}"].shape[1]
        num_heads[1] = state_dict[f"encoderlayer_1.{num_heads_suffix}"].shape[1]
        num_heads[2] = state_dict[f"encoderlayer_2.{num_heads_suffix}"].shape[1]
        num_heads[3] = state_dict[f"encoderlayer_3.{num_heads_suffix}"].shape[1]
        num_heads[4] = state_dict[f"conv.{num_heads_suffix}"].shape[1]
        num_heads[5] = state_dict[f"decoderlayer_0.{num_heads_suffix}"].shape[1]
        num_heads[6] = state_dict[f"decoderlayer_1.{num_heads_suffix}"].shape[1]
        num_heads[7] = state_dict[f"decoderlayer_2.{num_heads_suffix}"].shape[1]
        num_heads[8] = state_dict[f"decoderlayer_3.{num_heads_suffix}"].shape[1]

        if "encoderlayer_0.blocks.0.attn.qkv.to_q.depthwise.weight" in state_dict:
            token_projection = "conv"
            qkv_bias = True  # cannot be deduced from state_dict
        else:
            token_projection = "linear"
            qkv_bias = "encoderlayer_0.blocks.0.attn.qkv.to_q.bias" in state_dict

        modulator = "decoderlayer_0.blocks.0.modulator.weight" in state_dict
        cross_modulator = "decoderlayer_0.blocks.0.cross_modulator.weight" in state_dict

        # size_temp = (2 * win_size - 1) ** 2
        size_temp = state_dict[
            "encoderlayer_0.blocks.0.attn.relative_position_bias_table"
        ].shape[0]
        win_size = (int(math.sqrt(size_temp)) + 1) // 2

        if "encoderlayer_0.blocks.0.mlp.fc1.weight" in state_dict:
            token_mlp = "mlp"  # or "ffn", doesn't matter
            mlp_ratio = (
                state_dict["encoderlayer_0.blocks.0.mlp.fc1.weight"].shape[0]
                / embed_dim
            )
        elif state_dict["encoderlayer_0.blocks.0.mlp.dwconv.0.weight"].shape[1] == 1:
            token_mlp = "leff"
            mlp_ratio = (
                state_dict["encoderlayer_0.blocks.0.mlp.linear1.0.weight"].shape[0]
                / embed_dim
            )
        else:
            token_mlp = "fastleff"
            mlp_ratio = (
                state_dict["encoderlayer_0.blocks.0.mlp.linear1.0.weight"].shape[0]
                / embed_dim
            )

        model = Uformer(
            img_size=img_size,
            in_chans=in_chans,
            dd_in=dd_in,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            win_size=win_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            token_projection=token_projection,
            token_mlp=token_mlp,
            shift_flag=shift_flag,
            modulator=modulator,
            cross_modulator=cross_modulator,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration",
            tags=[],
            supports_half=False,  # Too much weirdness to support this at the moment
            supports_bfloat16=True,
            scale=1,
            input_channels=dd_in,
            output_channels=dd_in,
            size_requirements=SizeRequirements(multiple_of=128, square=True),
        )


__all__ = ["UformerArch", "Uformer"]
