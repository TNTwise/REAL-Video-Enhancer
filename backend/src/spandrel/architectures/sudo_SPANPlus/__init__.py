from typing_extensions import override

from ...util import KeyCondition, get_scale_and_output_channels

from ...__helpers.model_descriptor import Architecture, ImageModelDescriptor, StateDict
from .__arch.sudo_SPANPlus import sudo_SPANPlus


class sudo_SPANPlusArch(Architecture[sudo_SPANPlus]):  # noqa: N801
    def __init__(self):
        super().__init__(
            id="sudo_SPANPlus",
            detect=KeyCondition.has_all(
                "feats.0.sk.weight",
                "feats.1.block_1.c1_r.sk.weight",
                "feats.1.conv_2.sk.weight",
                "feats.1.conv_2.eval_conv.weight",
                "feats.1.conv_cat.weight",
                "upsampler.end_conv.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[sudo_SPANPlus]:
        # default values
        num_in_ch: int = 3
        num_out_ch: int = 3
        blocks: list[int] = [4]
        feature_channels: int = 64
        upscale: int = 2
        drop_rate: float = 0.0

        num_in_ch = 3
        num_out_ch = 3
        blocks = [4]
        feature_channels = state_dict["feats.0.conv.2.weight"].shape[
            0
        ]  # maybe this will work
        upscale = 2
        drop_rate = 0.0
        """upscale, num_out_ch = get_scale_and_output_channels(
            state_dict["upsampler.end_conv.weight"].shape[0],
            num_in_ch,
        )"""
        upscale = 2
        num_out_ch = 3

        model = sudo_SPANPlus(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            blocks=blocks,
            feature_channels=feature_channels,
            upscale=upscale,
            drop_rate=drop_rate,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture="sudo_SPANPlus",
            purpose="SR",
            tags=[],
            supports_half=True,
            supports_bfloat16=True,
            scale=upscale,  # TODO: fix me
            input_channels=num_in_ch,  # TODO: fix me
            output_channels=num_out_ch,  # TODO: fix me
        )


__all__ = ["sudo_SPANPlusArch", "sudo_SPANPlus"]
