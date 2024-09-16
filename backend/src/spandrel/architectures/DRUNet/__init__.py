from __future__ import annotations

import torch
from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.network_unet import DRUNet


class DRUNetArch(Architecture[DRUNet]):
    def __init__(self) -> None:
        super().__init__(
            id="DRUNet",
            detect=KeyCondition.has_all(
                "m_head.weight",
                "m_down1.0.res.0.weight",
                "m_down1.0.res.2.weight",
                "m_down2.0.res.0.weight",
                "m_down3.0.res.0.weight",
                "m_body.0.res.0.weight",
                "m_body.0.res.2.weight",
                "m_up3.2.res.0.weight",
                "m_up3.2.res.2.weight",
                "m_up2.2.res.0.weight",
                "m_up1.2.res.0.weight",
                "m_tail.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[DRUNet]:
        # in_nc = 1
        # out_nc = 1
        nc = [64, 128, 256, 512]
        # nb = 4
        act_mode = "R"  # this value is assumed
        downsample_mode = "strideconv"
        upsample_mode = "convtranspose"

        in_nc = state_dict["m_head.weight"].shape[1]
        out_nc = state_dict["m_tail.weight"].shape[0]

        nb = get_seq_len(state_dict, "m_body")

        nc[0] = state_dict["m_head.weight"].shape[0]
        nc[1] = state_dict["m_down2.0.res.0.weight"].shape[0]
        nc[2] = state_dict["m_down3.0.res.0.weight"].shape[0]
        nc[3] = state_dict["m_body.0.res.0.weight"].shape[0]

        if f"m_down1.{nb}.weight" in state_dict:
            downsample_mode = "strideconv"
        else:
            # avgpool and maxpool have the same state dict
            downsample_mode = "avgpool"

        if "m_up3.1.weight" in state_dict:
            upsample_mode = "upconv"
        elif state_dict["m_up3.0.weight"].shape[2] == 3:
            upsample_mode = "pixelshuffle"
        else:
            upsample_mode = "convtranspose"

        model = DRUNet(
            in_nc=in_nc,
            out_nc=out_nc,
            nc=nc,
            nb=nb,
            act_mode=act_mode,
            downsample_mode=downsample_mode,
            upsample_mode=upsample_mode,
        )

        def call(model: DRUNet, image: torch.Tensor) -> torch.Tensor:
            _, _, H, W = image.shape  # noqa: N806

            noise_level = 15 / 255  # default from repo
            noise_map = torch.zeros(1, 1, H, W).to(image) + noise_level

            return model(torch.cat([image, noise_map], dim=1))

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration",
            tags=[f"{nb}nb"],
            supports_half=False,  # TODO: verify
            supports_bfloat16=True,
            scale=1,
            input_channels=in_nc - 1,  # one channel is generated
            output_channels=out_nc,
            size_requirements=SizeRequirements(multiple_of=8),
            call_fn=call,
        )


__all__ = ["DRUNetArch", "DRUNet"]
