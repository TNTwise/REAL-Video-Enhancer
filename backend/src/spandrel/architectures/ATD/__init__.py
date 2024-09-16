import math

from typing_extensions import override

from spandrel.util import KeyCondition, get_pixelshuffle_params, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.atd_arch import ATD


class ATDArch(Architecture[ATD]):
    def __init__(
        self,
    ) -> None:
        super().__init__(
            id="ATD",
            name="ATD",
            detect=KeyCondition.has_all(
                "relative_position_index_SA",
                "conv_first.weight",
                "conv_first.bias",
                "layers.0.residual_group.td",
                "layers.0.residual_group.layers.0.sigma",
                "layers.0.residual_group.layers.0.norm1.weight",
                "layers.0.residual_group.layers.0.norm1.bias",
                "layers.0.residual_group.layers.0.norm2.weight",
                "layers.0.residual_group.layers.0.norm2.bias",
                "layers.0.residual_group.layers.0.norm3.weight",
                "layers.0.residual_group.layers.0.norm3.bias",
                "layers.0.residual_group.layers.0.wqkv.weight",
                "layers.0.residual_group.layers.0.attn_win.relative_position_bias_table",
                "layers.0.residual_group.layers.0.attn_win.proj.weight",
                "layers.0.residual_group.layers.0.attn_win.proj.bias",
                "layers.0.residual_group.layers.0.attn_atd.scale",
                "layers.0.residual_group.layers.0.attn_atd.wq.weight",
                "layers.0.residual_group.layers.0.attn_atd.wk.weight",
                "layers.0.residual_group.layers.0.attn_atd.wv.weight",
                "layers.0.residual_group.layers.0.attn_aca.logit_scale",
                "layers.0.residual_group.layers.0.attn_aca.proj.weight",
                "layers.0.residual_group.layers.0.convffn.fc1.weight",
                "layers.0.residual_group.layers.0.convffn.fc1.bias",
                "layers.0.residual_group.layers.0.convffn.dwconv.depthwise_conv.0.weight",
                "layers.0.residual_group.layers.0.convffn.dwconv.depthwise_conv.0.bias",
                "layers.0.residual_group.layers.0.convffn.fc2.weight",
                "layers.0.residual_group.layers.0.convffn.fc2.bias",
                "norm.weight",
                "norm.bias",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[ATD]:
        img_size = 64  # cannot be deduced from state dict
        patch_size = 1  # cannot be deduced from state dict
        in_chans = 3
        embed_dim = 90
        depths = (6, 6, 6, 6)
        num_heads = (6, 6, 6, 6)
        window_size = 8
        category_size = 256  # cannot be deduced from state dict
        num_tokens = 64
        reducted_dim = 4
        convffn_kernel_size = 5
        mlp_ratio = 2.0
        qkv_bias = True
        ape = False
        patch_norm = True
        upscale = 1
        img_range = 1.0  # cannot be deduced from state dict
        upsampler = ""
        resi_connection = "1conv"
        norm = True

        in_chans = state_dict["conv_first.weight"].shape[1]
        embed_dim = state_dict["conv_first.weight"].shape[0]

        window_size = math.isqrt(state_dict["relative_position_index_SA"].shape[0])

        num_layers = get_seq_len(state_dict, "layers")
        depths = [6] * num_layers
        num_heads = [6] * num_layers
        for i in range(num_layers):
            depths[i] = get_seq_len(state_dict, f"layers.{i}.residual_group.layers")
            num_heads[i] = state_dict[
                f"layers.{i}.residual_group.layers.0.attn_win.relative_position_bias_table"
            ].shape[1]

        num_tokens = state_dict[
            "layers.0.residual_group.layers.0.attn_atd.scale"
        ].shape[0]
        reducted_dim = state_dict[
            "layers.0.residual_group.layers.0.attn_atd.wq.weight"
        ].shape[0]
        convffn_kernel_size = state_dict[
            "layers.0.residual_group.layers.0.convffn.dwconv.depthwise_conv.0.weight"
        ].shape[2]
        mlp_ratio = (
            state_dict["layers.0.residual_group.layers.0.convffn.fc1.weight"].shape[0]
            / embed_dim
        )
        qkv_bias = "layers.0.residual_group.layers.0.wqkv.bias" in state_dict
        ape = "absolute_pos_embed" in state_dict
        patch_norm = "patch_embed.norm.weight" in state_dict

        resi_connection = "1conv" if "layers.0.conv.weight" in state_dict else "3conv"

        if "conv_up1.weight" in state_dict:
            upsampler = "nearest+conv"
            upscale = 4
        elif "conv_before_upsample.0.weight" in state_dict:
            upsampler = "pixelshuffle"
            upscale, _ = get_pixelshuffle_params(state_dict, "upsample")
        elif "conv_last.weight" in state_dict:
            upsampler = ""
            upscale = 1
        else:
            upsampler = "pixelshuffledirect"
            upscale = math.isqrt(state_dict["upsample.0.weight"].shape[0] // in_chans)

        norm = "no_norm" not in state_dict

        is_light = upsampler == "pixelshuffledirect" and embed_dim == 48
        # use a heuristic for category_size
        category_size = 128 if is_light else 256

        tags = [f"{embed_dim}dim", f"{window_size}w", f"{category_size}cat"]
        if is_light:
            tags.insert(0, "light")

        model = ATD(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            category_size=category_size,
            num_tokens=num_tokens,
            reducted_dim=reducted_dim,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            ape=ape,
            patch_norm=patch_norm,
            upscale=upscale,
            img_range=img_range,
            upsampler=upsampler,
            resi_connection=resi_connection,
            norm=norm,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if upscale == 1 else "SR",
            tags=tags,
            supports_half=False,  # TODO: verify
            supports_bfloat16=True,
            scale=upscale,
            input_channels=in_chans,
            output_channels=in_chans,
            size_requirements=SizeRequirements(minimum=8),
        )


__all__ = ["ATDArch", "ATD"]
