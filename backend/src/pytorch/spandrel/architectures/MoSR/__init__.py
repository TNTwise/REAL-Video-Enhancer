import math

from typing_extensions import override

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    StateDict,
)
from ...util import KeyCondition, get_seq_len
from .__arch.mosr_arch import MoSR


class MoSRArch(Architecture[MoSR]):
    def __init__(self) -> None:
        super().__init__(
            id='MoSR',
            detect=KeyCondition.has_all(
                'gblocks.0.weight',
                'gblocks.0.bias',
                'gblocks.1.norm.weight',
                'gblocks.1.norm.bias',
                'gblocks.1.fc1.weight',
                'gblocks.1.fc1.bias',
                'gblocks.1.conv.weight',
                'gblocks.1.conv.bias',
                'gblocks.1.fc2.weight',
                'gblocks.1.fc2.bias',
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[MoSR]:
        # default values
        in_ch = 3
        out_ch = 3
        upscale = 4
        n_block = 24
        dim = 64
        upsampler = 'ps'  # "ps" "dys", "gps"
        drop_path = 0.0
        kernel_size = 7
        expansion_ratio = 1.5
        conv_ratio = 1.0

        n_block = get_seq_len(state_dict, 'gblocks') - 6
        in_ch = state_dict['gblocks.0.weight'].shape[1]
        dim = state_dict['gblocks.0.weight'].shape[0]

        # Calculate expansion ratio and convolution ratio
        expansion_ratio = (
            state_dict['gblocks.1.fc1.weight'].shape[0]
            / state_dict['gblocks.1.fc1.weight'].shape[1]
        ) / 2
        conv_ratio = state_dict['gblocks.1.conv.weight'].shape[0] / dim
        kernel_size = state_dict['gblocks.1.conv.weight'].shape[2]
        # Determine upsampler type and calculate upscale
        if 'upsampler.init_pos' in state_dict:
            upsampler = 'dys'
            out_ch = state_dict['upsampler.end_conv.weight'].shape[0]
            upscale = math.isqrt(
                state_dict['upsampler.offset.weight'].shape[0] // 8
            )
        elif 'upsampler.in_to_k.weight' in state_dict:
            upsampler = 'gps'
            out_ch = in_ch
            upscale = math.isqrt(
                state_dict['upsampler.in_to_k.weight'].shape[0] // 8 // out_ch
            )
        else:
            upsampler = 'ps'
            out_ch = in_ch
            upscale = math.isqrt(
                state_dict['upsampler.0.weight'].shape[0] // out_ch
            )

        model = MoSR(
            in_ch=in_ch,
            out_ch=out_ch,
            upscale=upscale,
            n_block=n_block,
            dim=dim,
            upsampler=upsampler,
            drop_path=drop_path,
            kernel_size=kernel_size,
            expansion_ratio=expansion_ratio,
            conv_ratio=conv_ratio,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose='SR',
            tags=[],
            supports_half=True,
            supports_bfloat16=True,
            scale=upscale,
            input_channels=in_ch,
            output_channels=out_ch,
        )


__all__ = ['MoSR', 'MoSRArch']
