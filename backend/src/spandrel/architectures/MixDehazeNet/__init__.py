from __future__ import annotations

from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    ModelTiling,
    SizeRequirements,
    StateDict,
)
from .__arch.MixDehazeNet import MixDehazeNet


class MixDehazeNetArch(Architecture[MixDehazeNet]):
    def __init__(self) -> None:
        super().__init__(
            id="MixDehazeNet",
            detect=KeyCondition.has_all(
                "patch_embed.proj.weight",
                "patch_embed.proj.bias",
                "layer1.blocks.0.norm1.weight",
                "layer1.blocks.0.norm1.running_mean",
                "layer1.blocks.0.norm1.running_var",
                "layer1.blocks.0.norm2.running_mean",
                "layer1.blocks.0.conv1.weight",
                "layer1.blocks.0.conv3_19.weight",
                "layer1.blocks.0.conv3_13.weight",
                "layer1.blocks.0.conv3_7.weight",
                "layer1.blocks.0.Wv.0.weight",
                "layer1.blocks.0.Wv.1.weight",
                "layer1.blocks.0.Wg.1.weight",
                "layer1.blocks.0.ca.1.weight",
                "layer1.blocks.0.ca.1.weight",
                "layer1.blocks.0.ca.3.weight",
                "layer1.blocks.0.pa.0.weight",
                "layer1.blocks.0.pa.2.weight",
                "layer1.blocks.0.mlp.0.weight",
                "layer1.blocks.0.mlp.2.weight",
                "layer1.blocks.0.mlp2.0.weight",
                "layer1.blocks.0.mlp2.2.weight",
                "patch_merge1.proj.weight",
                "skip1.weight",
                "layer2.blocks.0.norm1.weight",
                "patch_merge2.proj.weight",
                "skip2.weight",
                "layer3.blocks.0.norm1.weight",
                "patch_split1.proj.0.weight",
                "fusion1.mlp.0.weight",
                "layer4.blocks.0.norm1.weight",
                "patch_split2.proj.0.weight",
                "fusion2.mlp.0.weight",
                "layer5.blocks.0.norm1.weight",
                "patch_unembed.proj.0.weight",
                "patch_unembed.proj.0.bias",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[MixDehazeNet]:
        in_chans = 3
        out_chans = 4
        embed_dims = [24, 48, 96, 48, 24]
        depths = [1, 1, 2, 1, 1]

        in_chans = state_dict["patch_embed.proj.weight"].shape[1]
        out_chans = state_dict["patch_unembed.proj.0.weight"].shape[0]

        embed_dims[0] = state_dict["layer1.blocks.0.conv1.weight"].shape[0]
        embed_dims[1] = state_dict["layer2.blocks.0.conv1.weight"].shape[0]
        embed_dims[2] = state_dict["layer3.blocks.0.conv1.weight"].shape[0]
        embed_dims[3] = state_dict["layer4.blocks.0.conv1.weight"].shape[0]
        embed_dims[4] = state_dict["layer5.blocks.0.conv1.weight"].shape[0]

        depths[0] = get_seq_len(state_dict, "layer1.blocks")
        depths[1] = get_seq_len(state_dict, "layer2.blocks")
        depths[2] = get_seq_len(state_dict, "layer3.blocks")
        depths[3] = get_seq_len(state_dict, "layer4.blocks")
        depths[4] = get_seq_len(state_dict, "layer5.blocks")

        model = MixDehazeNet(
            in_chans=in_chans,
            out_chans=out_chans,
            embed_dims=embed_dims,
            depths=depths,
        )

        tags = []
        if depths == [1, 1, 2, 1, 1]:
            tags.append("tiny")
        if depths == [2, 2, 4, 2, 2]:
            tags.append("small")
        if depths == [4, 4, 8, 4, 4]:
            tags.append("big")
        if depths == [8, 8, 16, 8, 8]:
            tags.append("large")

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration",
            tags=tags,
            supports_half=False,  # TODO: Test this
            supports_bfloat16=True,
            scale=1,
            input_channels=in_chans,
            output_channels=in_chans,
            size_requirements=SizeRequirements(minimum=40),
            tiling=ModelTiling.DISCOURAGED,
            call_fn=lambda model, image: model(image) * 0.5 + 0.5,
        )


__all__ = ["MixDehazeNetArch", "MixDehazeNet"]
