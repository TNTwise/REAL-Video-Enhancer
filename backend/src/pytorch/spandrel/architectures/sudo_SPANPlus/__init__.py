from sympy import false
from typing_extensions import override

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    StateDict,
)
from ...util import KeyCondition, get_scale_and_output_channels
from .__arch.sudo_SPANPlus import sudo_SPANPlus


class sudo_SPANPlusArch(Architecture[sudo_SPANPlus]):  # noqa: N801
    def __init__(self):
        super().__init__(
            id='sudo_SPANPlus',
            detect=KeyCondition.has_all(
                'feats.0.sk.weight',
                'feats.1.block_1.c1_r.sk.weight',
                'feats.1.conv_2.sk.weight',
                'feats.1.conv_2.eval_conv.weight',
                'feats.1.conv_cat.weight',
                'dynamic.kernels_weights',
                'dynamic.attention.to_scores.0.weight',
            ),
        )

    @override
    def load(
        self, state_dict: StateDict
    ) -> ImageModelDescriptor[sudo_SPANPlus]:
        # default values
        num_in_ch: int = 3
        num_out_ch: int = 3
        blocks: list[int] = [4]
        feature_channels: int = 64
        upscale: int = 2
        drop_rate: float = 0.0

        num_in_ch = state_dict['feats.0.conv.0.weight'].shape[1]
        feature_channels = state_dict['feats.0.conv.2.weight'].shape[
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

        downsample = num_in_ch != num_out_ch

        model = sudo_SPANPlus(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            blocks=blocks,
            feature_channels=feature_channels,
            upscale=upscale,
            drop_rate=drop_rate,
            downsample=downsample,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture='sudo_SPANPlus',
            purpose='SR',
            tags=[],
            supports_half=True,
            supports_bfloat16=True,
            scale=upscale,  # TODO: fix me
            input_channels=3,  # TODO: fix me
            output_channels=3,  # TODO: fix me
        )


__all__ = ['get_scale_and_output_channels', 'sudo_SPANPlus', 'sudo_SPANPlusArch']
