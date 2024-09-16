import torch
from typing_extensions import override

from spandrel.util import KeyCondition, get_scale_and_output_channels

from ...__helpers.model_descriptor import Architecture, ImageModelDescriptor, StateDict
from .__arch.span import SPAN


class SPANArch(Architecture[SPAN]):
    def __init__(self) -> None:
        super().__init__(
            id="SPAN",
            detect=KeyCondition.has_all(
                "conv_1.sk.weight",
                "block_1.c1_r.sk.weight",
                "block_1.c1_r.eval_conv.weight",
                "block_1.c3_r.eval_conv.weight",
                "conv_cat.weight",
                "conv_2.sk.weight",
                "conv_2.eval_conv.weight",
                "upsampler.0.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[SPAN]:
        num_in_ch: int = 3
        num_out_ch: int = 3
        feature_channels: int = 48
        upscale: int = 4
        bias = True  # unused internally
        norm = True
        img_range = 255.0  # cannot be deduced from state_dict
        rgb_mean = (0.4488, 0.4371, 0.4040)  # cannot be deduced from state_dict

        num_in_ch = state_dict["conv_1.sk.weight"].shape[1]
        feature_channels = state_dict["conv_1.sk.weight"].shape[0]

        # pixelshuffel shenanigans
        upscale, num_out_ch = get_scale_and_output_channels(
            state_dict["upsampler.0.weight"].shape[0],
            num_in_ch,
        )

        # norm
        if "no_norm" in state_dict:
            norm = False
            state_dict["no_norm"] = torch.zeros(1)

        model = SPAN(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            feature_channels=feature_channels,
            upscale=upscale,
            bias=bias,
            norm=norm,
            img_range=img_range,
            rgb_mean=rgb_mean,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if upscale == 1 else "SR",
            tags=[f"{feature_channels}nf"],
            supports_half=True,
            supports_bfloat16=True,
            scale=upscale,
            input_channels=num_in_ch,
            output_channels=num_out_ch,
        )


__all__ = ["SPANArch", "SPAN"]
