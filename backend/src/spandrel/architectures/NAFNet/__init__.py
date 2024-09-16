from __future__ import annotations

from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import Architecture, ImageModelDescriptor, StateDict
from .__arch.NAFNet_arch import NAFNet


class NAFNetArch(Architecture[NAFNet]):
    def __init__(self) -> None:
        super().__init__(
            id="NAFNet",
            detect=KeyCondition.has_all(
                "intro.weight",
                "ending.weight",
                "ups.0.0.weight",
                "downs.0.weight",
                "middle_blks.0.beta",
                "middle_blks.0.gamma",
                "middle_blks.0.conv1.weight",
                "middle_blks.0.conv2.weight",
                "middle_blks.0.conv3.weight",
                "middle_blks.0.sca.1.weight",
                "middle_blks.0.conv4.weight",
                "middle_blks.0.conv5.weight",
                "middle_blks.0.norm1.weight",
                "middle_blks.0.norm2.weight",
                "encoders.0.0.beta",
                "encoders.0.0.gamma",
                "decoders.0.0.beta",
                "decoders.0.0.gamma",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[NAFNet]:
        # default values
        img_channel: int = 3
        width: int = 16
        middle_blk_num: int = 1
        enc_blk_nums: list[int] = []
        dec_blk_nums: list[int] = []

        img_channel = state_dict["intro.weight"].shape[1]
        width = state_dict["intro.weight"].shape[0]
        middle_blk_num = get_seq_len(state_dict, "middle_blks")
        for i in range(get_seq_len(state_dict, "encoders")):
            enc_blk_nums.append(get_seq_len(state_dict, f"encoders.{i}"))
        for i in range(get_seq_len(state_dict, "decoders")):
            dec_blk_nums.append(get_seq_len(state_dict, f"decoders.{i}"))

        model = NAFNet(
            img_channel=img_channel,
            width=width,
            middle_blk_num=middle_blk_num,
            enc_blk_nums=enc_blk_nums,
            dec_blk_nums=dec_blk_nums,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration",
            tags=[f"{width}w"],
            supports_half=False,  # TODO: Test this
            supports_bfloat16=True,
            scale=1,
            input_channels=img_channel,
            output_channels=img_channel,
        )


__all__ = ["NAFNetArch", "NAFNet"]
