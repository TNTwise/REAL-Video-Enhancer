from typing_extensions import override

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    StateDict,
)
from ...util import KeyCondition, get_scale_and_output_channels, get_seq_len
from .__arch.Sebica import RTSRSebica as Sebica


class SebicaArch(Architecture[Sebica]):
    def __init__(
        self,
    ) -> None:
        super().__init__(
            id='Sebica',
            name='Sebica',
            detect=KeyCondition.has_all(
                'head.1.weight',
                'body.0.pre_mixer.conv.0.weight',
                'body.0.pre_mixer.conv.1.weight',
                'body.0.post_mixer.ffn.0.weight',
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[Sebica]:
        num_in_ch = 3
        num_out_ch = 3
        scale = 4
        num_feat = 16
        attn_blocks = 6
        state = state_dict

        attn_blocks = get_seq_len(state, 'body')
        num_in_ch = state['head.0.weight'].shape[1]

        num_feat = state['head.1.weight'].shape[0]
        pixelshuffle_shape = state_dict['tail.0.weight'].shape[0]

        scale, num_out_ch = get_scale_and_output_channels(
            pixelshuffle_shape, num_in_ch
        )
        model = Sebica(
            sr_rate=scale,
            num_feat=num_feat,
            attn_blocks=attn_blocks,
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
        )

        return ImageModelDescriptor(
            model,
            state,
            architecture=self,
            purpose='Restoration' if scale == 1 else 'SR',
            tags=[f'{num_feat}nf'],
            supports_half=True,
            supports_bfloat16=True,
            scale=scale,
            input_channels=num_in_ch,
            output_channels=num_out_ch,
        )


__all__ = ['Sebica', 'SebicaArch']
