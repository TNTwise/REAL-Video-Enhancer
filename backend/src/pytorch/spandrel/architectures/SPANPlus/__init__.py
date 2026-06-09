from typing_extensions import override

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    StateDict,
)
from ...util import KeyCondition, get_seq_len
from .__arch.spanplus import SPANPlus


class SPANPlusArch(Architecture[SPANPlus]):
    def __init__(self) -> None:
        super().__init__(
            id="SPANPlus",
            detect=KeyCondition.has_all(
                "feats.0.eval_conv.weight",
                "feats.1.block_1.c1_r.sk.weight",
                "feats.1.block_1.c1_r.conv.0.weight",
                "feats.1.block_1.c1_r.eval_conv.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[SPANPlus]:
        num_in_ch: int = 3
        num_out_ch: int = 3
        feature_channels: int = 48
        upscale: int = 4
        n_feats = get_seq_len(state_dict, "feats") - 1
        blocks = [
            get_seq_len(state_dict, f"feats.{n_feat + 1}.block_n")
            for n_feat in range(n_feats)
        ]
        num_in_ch = state_dict["feats.0.eval_conv.weight"].shape[1]
        feature_channels = state_dict["feats.0.eval_conv.weight"].shape[0]
        if "upsampler.0.weight" in state_dict:
            upsampler = "ps"
            num_out_ch = num_in_ch
            upscale = int(
                (state_dict["upsampler.0.weight"].shape[0] / num_in_ch) ** 0.5
            )
        else:
            upsampler = "dys"
            num_out_ch = state_dict["upsampler.end_conv.weight"].shape[0]
            upscale = int((state_dict["upsampler.offset.weight"].shape[0] // 8) ** 0.5)

        model = SPANPlus(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            feature_channels=feature_channels,
            upscale=upscale,
            blocks=blocks,
            upsampler=upsampler,
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


__all__ = ["SPANPlus", "SPANPlusArch"]
