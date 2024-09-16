import math

from typing_extensions import override

from spandrel.util import KeyCondition, get_pixelshuffle_params, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.HAT import HAT


def _get_overlap_ratio(window_size: int, with_overlap: int) -> float:
    # What we know:
    #   with_overlap = int(window_size + window_size * overlap_ratio)
    #
    # The issue is that this relationship doesn't uniquely define overlap_ratio. E.g.
    # for window_size=7, overlap_ratio=0.5 and overlap_ratio=0.51 both result in
    # with_overlap=10. So to get "nice" ratios, we will first try out "nice" numbers
    # before falling back to the general formula.

    nice_numbers = [0, 1, 0.5, 0.25, 0.75, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    for ratio in nice_numbers:
        if int(window_size + window_size * ratio) == with_overlap:
            return ratio

    # calculate the ratio and add a little something to account for rounding errors
    return (with_overlap - window_size) / window_size + 0.01


def _inv_int_div(a: int, c: int) -> float:
    """
    Returns a number `b` such that `a // b == c`.
    """
    b_float = a / c

    if b_float.is_integer():
        return int(b_float)
    if c == a // math.ceil(b_float):
        return math.ceil(b_float)
    if c == a // math.floor(b_float):
        return math.floor(b_float)

    # account for rounding errors
    if c == a // b_float:
        return b_float
    if c == a // (b_float - 0.01):
        return b_float - 0.01
    if c == a // (b_float + 0.01):
        return b_float + 0.01

    raise ValueError(f"Could not find a number b such that a // b == c. a={a}, c={c}")


class HATArch(Architecture[HAT]):
    def __init__(self) -> None:
        super().__init__(
            id="HAT",
            detect=KeyCondition.has_all(
                "relative_position_index_SA",
                "conv_first.weight",
                "layers.0.residual_group.blocks.0.norm1.weight",
                "layers.0.residual_group.blocks.0.conv_block.cab.0.weight",
                "layers.0.residual_group.blocks.0.conv_block.cab.2.weight",
                "layers.0.residual_group.blocks.0.conv_block.cab.3.attention.1.weight",
                "layers.0.residual_group.blocks.0.conv_block.cab.3.attention.3.weight",
                "layers.0.residual_group.blocks.0.mlp.fc1.bias",
                "layers.0.residual_group.blocks.0.mlp.fc2.weight",
                "layers.0.residual_group.overlap_attn.relative_position_bias_table",
                "layers.0.residual_group.overlap_attn.qkv.weight",
                "layers.0.residual_group.overlap_attn.proj.weight",
                "layers.0.residual_group.overlap_attn.mlp.fc1.weight",
                "layers.0.residual_group.overlap_attn.mlp.fc2.weight",
                "conv_last.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[HAT]:
        img_size = 64
        patch_size = 1
        in_chans = 3
        embed_dim = 96
        depths = (6, 6, 6, 6)
        num_heads = (6, 6, 6, 6)
        window_size = 7
        compress_ratio = 3
        squeeze_factor = 30
        conv_scale = 0.01  # cannot be deduced from state dict
        overlap_ratio = 0.5
        mlp_ratio = 4.0
        qkv_bias = True
        qk_scale = None  # cannot be deduced from state dict
        drop_rate = 0.0  # cannot be deduced from state dict
        attn_drop_rate = 0.0  # cannot be deduced from state dict
        drop_path_rate = 0.1  # cannot be deduced from state dict
        ape = False
        patch_norm = True
        upscale = 2
        img_range = 1.0  # cannot be deduced from state dict
        upsampler = "pixelshuffle"  # it's the only possible value
        resi_connection = "1conv"
        num_feat = 64

        in_chans = state_dict["conv_first.weight"].shape[1]
        embed_dim = state_dict["conv_first.weight"].shape[0]

        num_feat = state_dict["conv_last.weight"].shape[1]
        upscale, _ = get_pixelshuffle_params(state_dict, "upsample", num_feat)

        window_size = int(math.sqrt(state_dict["relative_position_index_SA"].shape[0]))
        overlap_ratio = _get_overlap_ratio(
            window_size,
            with_overlap=int(
                math.sqrt(state_dict["relative_position_index_OCA"].shape[1])
            ),
        )

        # num_layers = len(depths)
        num_layers = get_seq_len(state_dict, "layers")
        depths = [
            get_seq_len(state_dict, f"layers.{i}.residual_group.blocks")
            for i in range(num_layers)
        ]
        num_heads = [
            state_dict[
                f"layers.{i}.residual_group.overlap_attn.relative_position_bias_table"
            ].shape[1]
            for i in range(num_layers)
        ]

        if "conv_after_body.weight" in state_dict:
            resi_connection = "1conv"
        else:
            # There is no way to decide whether it's "identity" or something else.
            # So we just assume it's identity.
            resi_connection = "identity"

        compress_ratio = _inv_int_div(
            embed_dim,
            state_dict[
                "layers.0.residual_group.blocks.0.conv_block.cab.0.weight"
            ].shape[0],
        )
        squeeze_factor = _inv_int_div(
            embed_dim,
            state_dict[
                "layers.0.residual_group.blocks.0.conv_block.cab.3.attention.1.weight"
            ].shape[0],
        )

        qkv_bias = "layers.0.residual_group.blocks.0.attn.qkv.bias" in state_dict
        patch_norm = "patch_embed.norm.weight" in state_dict
        ape = "absolute_pos_embed" in state_dict

        # mlp_hidden_dim = int(embed_dim * mlp_ratio)
        mlp_hidden_dim = int(
            state_dict["layers.0.residual_group.blocks.0.mlp.fc1.weight"].shape[0]
        )
        mlp_ratio = mlp_hidden_dim / embed_dim

        # img_size and patch_size are linked to each other and not always stored in the
        # state dict. If it isn't stored, then there is no way to deduce it.
        if "absolute_pos_embed" in state_dict:
            # patches_resolution = img_size // patch_size
            # num_patches = patches_resolution ** 2
            num_patches = state_dict["absolute_pos_embed"].shape[1]
            patches_resolution = int(math.sqrt(num_patches))
            # we'll just assume that the patch size is 1
            patch_size = 1
            img_size = patches_resolution

        model = HAT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            conv_scale=conv_scale,
            overlap_ratio=overlap_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            ape=ape,
            patch_norm=patch_norm,
            upscale=upscale,
            img_range=img_range,
            upsampler=upsampler,
            resi_connection=resi_connection,
            num_feat=num_feat,
        )

        if len(depths) < 9:
            size_tag = "small" if compress_ratio > 4 else "medium"
        else:
            size_tag = "large"
        tags = [
            size_tag,
            f"s{img_size}w{window_size}",
            f"{num_feat}nf",
            f"{embed_dim}dim",
            f"{resi_connection}",
        ]

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if upscale == 1 else "SR",
            tags=tags,
            supports_half=False,
            supports_bfloat16=True,
            scale=upscale,
            input_channels=in_chans,
            output_channels=in_chans,
            size_requirements=SizeRequirements(minimum=16),
        )


__all__ = ["HATArch", "HAT"]
