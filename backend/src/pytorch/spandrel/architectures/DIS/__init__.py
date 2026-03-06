import math

from typing_extensions import override

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    StateDict,
)
from ...util import KeyCondition, get_seq_len
from .__arch.dis import DIS


class DISArch(Architecture[DIS]):
    def __init__(self) -> None:
        super().__init__(
            id='DIS',
            name='Direct Image Supersampling',
            detect=KeyCondition.has_all(
                'head.weight',
                'head_act.weight',
                'fusion.weight',
                'tail.weight',
                KeyCondition.has_any(
                    'body.0.conv1.weight',
                    'body.0.dw_conv.depthwise.weight',
                ),
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[DIS]:
        in_channels = state_dict['head.weight'].shape[1]
        num_features = state_dict['head.weight'].shape[0]
        out_channels = state_dict['tail.weight'].shape[0]
        num_blocks = get_seq_len(state_dict, 'body')
        use_depthwise = 'body.0.dw_conv.depthwise.weight' in state_dict

        if 'upsampler.0.conv.weight' in state_dict:
            scale = 4
        elif 'upsampler.conv.weight' in state_dict:
            upscale_sq = state_dict['upsampler.conv.weight'].shape[0] // num_features
            scale = math.isqrt(upscale_sq)
        else:
            scale = 1

        model = DIS(
            in_channels=in_channels,
            out_channels=out_channels,
            num_features=num_features,
            num_blocks=num_blocks,
            scale=scale,
            use_depthwise=use_depthwise,
        )

        tags = [f'{num_features}nf', f'{num_blocks}nb']
        if use_depthwise:
            tags.append('depthwise')

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose='Restoration' if scale == 1 else 'SR',
            tags=tags,
            supports_half=True,
            supports_bfloat16=True,
            scale=scale,
            input_channels=in_channels,
            output_channels=out_channels,
        )


__all__ = ['DIS', 'DISArch']
