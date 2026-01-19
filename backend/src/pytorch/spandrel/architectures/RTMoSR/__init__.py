import math

from typing_extensions import override

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    StateDict,
)
from ...util import KeyCondition, get_seq_len
from .__arch.RTMoSR import RTMoSR


class RTMoSRArch(Architecture[RTMoSR]):
    def __init__(self):
        super().__init__(
            id="RTMoSR",
            name="Real Time MoSR",
            detect=KeyCondition.has_all(
                "body.0.fc1.alpha",
                "to_img.0.alpha",
                "to_img.0.conv1.k0",
                "to_img.0.conv2.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[RTMoSR]:
        in_nc = 3
        out_nc = 3
        unshuffle = True
        """if "to_feat.1.weight" in state_dict:
            unshuffle = True
            scale = math.isqrt(state_dict["to_feat.1.weight"].shape[1] // 3)
            dim = state_dict["to_feat.1.weight"].shape[0]
        else:
            scale = math.isqrt(state_dict["to_img.0.weight"].shape[0] // 3)
            dim = state_dict["to_feat.weight"].shape[0]
        ffn = state_dict["body.0.fc1.conv1.weight"].shape[0] / dim / 2"""
        n_blocks = get_seq_len(state_dict, "body")
        upscaling_factor = int(math.sqrt(state_dict["to_img.0.conv1.b1"].shape[0] / 3))
        ffn_expansion = state_dict["body.0.fc1.conv1.k0"].shape[0] / 128
        dim = state_dict["body.0.norm.scale"].shape[0]
        unshuffle = "to_feat.1.alpha" in state_dict.keys()
        if unshuffle:
            w = state_dict["to_feat.1.conv1.k0"].shape[1]
            if w == 48:
                upscaling_factor = 1
            else:
                upscaling_factor = math.isqrt(
                    state_dict["to_feat.1.weight"].shape[1] // 3
                )
        model = RTMoSR(
            scale=upscaling_factor,
            dim=dim,
            ffn_expansion=ffn_expansion,
            n_blocks=n_blocks,
            unshuffle_mod=unshuffle,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if upscaling_factor == 1 else "SR",
            tags=[f"{dim}nf", f"{n_blocks}nc"],
            supports_half=True,
            supports_bfloat16=True,
            scale=upscaling_factor,
            input_channels=in_nc,
            output_channels=out_nc,
        )


__all__ = ["RTMoSRArch", "RTMoSR"]
