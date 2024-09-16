from __future__ import annotations

from typing import Literal

from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.mmrealsr_arch import MMRRDBNet_test as MMRealSR


def _get_in_ch_and_scale(combined: int, num_out_ch: int) -> tuple[int, int]:
    # num_in_ch and scale are linked in the original code like this:
    #   if scale == 2:
    #       num_in_ch = num_in_ch * 4
    #   elif scale == 1:
    #       num_in_ch = num_in_ch * 16
    # It is not possible to read the original values for num_in_ch and scale, so we
    # have to deduce them from the combined value. Luckily, only scale=1,2,4 are supported.
    # We'll also assume that num_in_ch == num_out_ch is very likely
    if combined in (num_out_ch, num_out_ch * 4, num_out_ch * 16):
        num_in_ch = num_out_ch
    elif combined % 3 == 0:
        num_in_ch = 3
    elif combined % 16 == 0 and combined > 16:
        num_in_ch = combined // 16
    elif combined % 4 != 0:
        num_in_ch = combined
    else:
        # it's impossible to differentiate between 2x RGBA vs 1x grayscale and
        # 4x RGBA vs 2x grayscale, so we assume that 2x models are unlikely
        if combined == 16:
            # 2x RGBA vs 1x grayscale
            num_in_ch = 1
        else:
            # 4x RGBA vs 2x grayscale
            num_in_ch = 4

    if num_in_ch * 1 == combined:
        scale = 4
    elif num_in_ch * 4 == combined:
        scale = 2
    elif num_in_ch * 16 == combined:
        scale = 1
    else:
        raise ValueError(
            f"Could not infer scale from num_in_ch={num_in_ch} and num_out_ch={num_out_ch}"
        )

    return num_in_ch, scale


def _get_num_feats_blocks_and_downscales(state_dict: StateDict):
    # This code is really complicated, but it's essentially a state machine.
    # To understand this code, you first need to understand the code in `DEResNet`
    # that generates body.
    num_feats: list[int] = []
    num_blocks: list[int] = []
    downscales: list[Literal[1, 2]] = []

    last_feats: int | None = None
    blocks: int = 0

    i = 0
    while True:
        if f"de_net.body.0.{i}.conv1.weight" in state_dict:
            # ResidualBlockNoBN
            feats = state_dict[f"de_net.body.0.{i}.conv1.weight"].shape[0]

            if last_feats is None:
                last_feats = feats
                blocks = 1
            elif feats == last_feats:
                blocks += 1
            else:
                num_feats.append(last_feats)
                num_blocks.append(blocks)
                downscales.append(1)

                last_feats = feats
                blocks = 1

        elif f"de_net.body.0.{i}.weight" in state_dict:
            # nn.Conv2d
            # Since downscales=2 is a superset of downscales=1, we cannot
            # differentiate	them and have to assume downscales=2

            assert last_feats is not None
            num_feats.append(last_feats)
            num_blocks.append(blocks)
            downscales.append(2)

            last_feats = None
            blocks = 0

        else:
            # end of body
            if last_feats is not None:
                num_feats.append(last_feats)
                num_blocks.append(blocks)
                downscales.append(1)
            break

        i += 1

    return num_feats, num_blocks, downscales


class MMRealSRArch(Architecture[MMRealSR]):
    def __init__(self) -> None:
        super().__init__(
            id="MMRealSR",
            detect=KeyCondition.has_all(
                "conv_first.weight",
                "conv_body.weight",
                "conv_up1.weight",
                "conv_up2.weight",
                "conv_hr.weight",
                "conv_last.weight",
                "body.0.rdb1.conv1.weight",
                "am_list.0.fc.0.weight",
                "de_net.conv_first.0.weight",
                "de_net.body.0.0.conv1.weight",
                "de_net.fc_degree.0.0.weight",
                "dd_embed.0.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[MMRealSR]:
        # default values
        num_in_ch = 3
        num_out_ch = 3
        scale = 4
        num_feat = 64
        num_block = 23
        num_grow_ch = 32
        num_degradation = 2
        degradation_degree_actv = "sigmoid"  # cannot be deduced from state_dict
        num_feats = [64, 128, 256, 512]
        num_blocks = [2, 2, 2, 2]
        downscales = [2, 2, 2, 1]

        num_out_ch = state_dict["conv_last.weight"].shape[0]
        num_feat = state_dict["conv_last.weight"].shape[1]

        combined = state_dict["conv_first.weight"].shape[1]
        num_in_ch, scale = _get_in_ch_and_scale(combined, num_out_ch)

        num_block = get_seq_len(state_dict, "body")
        num_grow_ch = state_dict["body.0.rdb1.conv1.weight"].shape[0]

        num_degradation = state_dict["dd_embed.0.weight"].shape[1]

        num_feats, num_blocks, downscales = _get_num_feats_blocks_and_downscales(
            state_dict
        )

        model = MMRealSR(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            scale=scale,
            num_feat=num_feat,
            num_block=num_block,
            num_grow_ch=num_grow_ch,
            num_degradation=num_degradation,
            degradation_degree_actv=degradation_degree_actv,
            num_feats=num_feats,
            num_blocks=num_blocks,
            downscales=downscales,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if scale == 1 else "SR",
            tags=[
                f"{num_feat}nf",
                f"{num_block}nb",
            ],
            supports_half=True,  # TODO: Test this
            supports_bfloat16=True,
            scale=scale,
            input_channels=num_in_ch,
            output_channels=num_out_ch,
            size_requirements=SizeRequirements(minimum=16),
            call_fn=lambda model, image: model(image)[0],
        )


__all__ = ["MMRealSRArch", "MMRealSR"]
