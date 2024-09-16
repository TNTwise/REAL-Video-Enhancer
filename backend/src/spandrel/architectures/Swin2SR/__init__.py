import math

from typing_extensions import override

from spandrel.util import KeyCondition, get_pixelshuffle_params, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.Swin2SR import Swin2SR


class Swin2SRArch(Architecture[Swin2SR]):
    def __init__(self) -> None:
        super().__init__(
            id="Swin2SR",
            detect=KeyCondition.has_all(
                "layers.0.residual_group.blocks.0.norm1.weight",
                "patch_embed.proj.weight",
                "conv_first.weight",
                "layers.0.residual_group.blocks.0.mlp.fc1.bias",
                "layers.0.residual_group.blocks.0.attn.relative_position_index",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[Swin2SR]:
        # Defaults
        img_size = 64
        patch_size = 1
        in_chans = 3
        embed_dim = 96
        depths = [6, 6, 6, 6]
        num_heads = [6, 6, 6, 6]
        window_size = 7
        mlp_ratio = 4.0
        qkv_bias = True
        drop_rate = 0.0  # cannot be deduced from state_dict
        attn_drop_rate = 0.0  # cannot be deduced from state_dict
        drop_path_rate = 0.1  # cannot be deduced from state_dict
        ape = False
        patch_norm = True
        use_checkpoint = False  # cannot be deduced from state_dict
        upscale = 2
        img_range = 1.0
        upsampler = ""
        resi_connection = "1conv"

        in_chans = state_dict["conv_first.weight"].shape[1]
        embed_dim = state_dict["conv_first.weight"].shape[0]
        patch_size = state_dict["patch_embed.proj.weight"].shape[2]

        ape = "absolute_pos_embed" in state_dict
        patch_norm = "patch_embed.norm.weight" in state_dict
        qkv_bias = "layers.0.residual_group.blocks.0.attn.q_bias" in state_dict

        # depths & num_heads
        num_layers = get_seq_len(state_dict, "layers")
        depths = [6] * num_layers
        num_heads = [6] * num_layers
        for i in range(num_layers):
            depths[i] = get_seq_len(state_dict, f"layers.{i}.residual_group.blocks")
            num_heads[i] = state_dict[
                f"layers.{i}.residual_group.blocks.0.attn.logit_scale"
            ].shape[0]

        mlp_ratio = float(
            state_dict["layers.0.residual_group.blocks.0.mlp.fc1.weight"].shape[0]
            / embed_dim
        )

        if "conv_after_body.0.weight" in state_dict:
            resi_connection = "3conv"
        elif "conv_after_body.weight" in state_dict:
            resi_connection = "1conv"
        else:
            raise ValueError("Unknown residual connection type")

        # upsampler
        if "conv_bicubic.weight" in state_dict:
            upsampler = "pixelshuffle_aux"
        elif "conv_hr.weight" in state_dict:
            upsampler = "nearest+conv"
        elif "conv_after_body_hf.weight" in state_dict:
            upsampler = "pixelshuffle_hf"
        elif "conv_before_upsample.0.weight" in state_dict:
            upsampler = "pixelshuffle"
        elif "upsample.0.weight" in state_dict:
            upsampler = "pixelshuffledirect"
        else:
            upsampler = ""

        if upsampler == "":
            upscale = 1
        elif upsampler == "nearest+conv":
            upscale = 4  # only supports 4x
        elif upsampler == "pixelshuffledirect":
            upscale = int(
                math.sqrt(state_dict["upsample.0.weight"].shape[0] // in_chans)
            )
        else:
            upscale, _ = get_pixelshuffle_params(state_dict, "upsample")

        window_size = int(
            math.sqrt(
                state_dict[
                    "layers.0.residual_group.blocks.0.attn.relative_position_index"
                ].shape[0]
            )
        )

        # Now for img_size... What we know:
        #   patches_resolution = img_size // patch_size
        #   if window_size > patches_resolution:
        #     attn_mask[0] = patches_resolution**2 // window_size**2
        if "layers.0.residual_group.blocks.1.attn_mask" in state_dict:
            attn_mask_0 = state_dict[
                "layers.0.residual_group.blocks.1.attn_mask"
            ].shape[0]
            patches_resolution = int(math.sqrt(attn_mask_0 * window_size * window_size))
            img_size = patches_resolution * patch_size
        else:
            # we only know that window_size <= patches_resolution
            # assume window_size == patches_resolution
            img_size = patch_size * window_size

            # if APE is enabled, we know that absolute_pos_embed[1] == patches_resolution**2
            if ape:
                patches_resolution = int(math.sqrt(state_dict["absolute_pos_embed"][1]))
                img_size = patches_resolution * patch_size

        # The JPEG models are the only ones with window-size 7, and they also use this range
        img_range = 255.0 if window_size == 7 else 1.0

        model = Swin2SR(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            upscale=upscale,
            img_range=img_range,
            upsampler=upsampler,
            resi_connection=resi_connection,
        )

        head_length = len(depths)  # type: ignore
        if head_length <= 4:
            size_tag = "small"
        elif head_length < 9:
            size_tag = "medium"
        else:
            size_tag = "large"
        tags = [
            size_tag,
            f"s{img_size}w{window_size}",
            f"{embed_dim}dim",
            f"{resi_connection}",
        ]

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if upscale == 1 else "SR",
            tags=tags,
            supports_half=False,  # Too much weirdness to support this at the moment
            supports_bfloat16=True,
            scale=upscale,
            input_channels=in_chans,
            output_channels=in_chans,
            size_requirements=SizeRequirements(minimum=16),
        )


__all__ = ["Swin2SRArch", "Swin2SR"]
