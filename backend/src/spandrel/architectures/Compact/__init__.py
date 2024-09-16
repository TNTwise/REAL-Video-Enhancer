from typing_extensions import override

from spandrel.util import KeyCondition, get_scale_and_output_channels, get_seq_len

from ...__helpers.model_descriptor import Architecture, ImageModelDescriptor, StateDict
from .__arch.SRVGG import SRVGGNetCompact as Compact


class CompactArch(Architecture[Compact]):
    def __init__(
        self,
    ) -> None:
        super().__init__(
            id="Compact",
            name="RealESRGAN Compact",
            detect=KeyCondition.has_all(
                "body.0.weight",
                "body.1.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[Compact]:
        state = state_dict

        highest_num = get_seq_len(state, "body") - 1

        in_nc = state["body.0.weight"].shape[1]
        num_feat = state["body.0.weight"].shape[0]
        num_conv = (highest_num - 2) // 2

        pixelshuffle_shape = state[f"body.{highest_num}.bias"].shape[0]
        scale, out_nc = get_scale_and_output_channels(pixelshuffle_shape, in_nc)

        model = Compact(
            num_in_ch=in_nc,
            num_out_ch=out_nc,
            num_feat=num_feat,
            num_conv=num_conv,
            upscale=scale,
        )

        return ImageModelDescriptor(
            model,
            state,
            architecture=self,
            purpose="Restoration" if scale == 1 else "SR",
            tags=[f"{num_feat}nf", f"{num_conv}nc"],
            supports_half=True,
            supports_bfloat16=True,
            scale=scale,
            input_channels=in_nc,
            output_channels=out_nc,
        )


__all__ = ["CompactArch", "Compact"]
