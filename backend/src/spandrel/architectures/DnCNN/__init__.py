from __future__ import annotations

from typing import Literal

import torch
from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.network_dncnn import DnCNN


class DnCNNArch(Architecture[DnCNN]):
    def __init__(self) -> None:
        super().__init__(
            id="DnCNN",
            detect=KeyCondition.has_all(
                "model.0.weight",
                "model.0.bias",
                "model.2.weight",
                "model.2.bias",
                KeyCondition.has_any(
                    KeyCondition.has_all(
                        # act_mode="R"
                        "model.4.weight",
                        "model.4.bias",
                        "model.6.weight",
                        "model.6.bias",
                        "model.8.weight",
                        "model.8.bias",
                        "model.10.weight",
                        "model.10.bias",
                        "model.12.weight",
                        "model.12.bias",
                        "model.14.weight",
                        "model.14.bias",
                    ),
                    KeyCondition.has_all(
                        # act_mode="BR"
                        "model.3.weight",
                        "model.3.bias",
                        "model.3.running_mean",
                        "model.3.running_var",
                        "model.5.weight",
                        "model.5.bias",
                        "model.6.weight",
                        "model.6.bias",
                        "model.6.running_mean",
                        "model.6.running_var",
                        "model.8.weight",
                        "model.8.bias",
                    ),
                ),
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[DnCNN]:
        # in_nc = 1
        # out_nc = 1
        # nc = 64
        # nb = 17
        # act_mode = "BR"
        mode: Literal["DnCNN", "FDnCNN"] = "DnCNN"

        in_nc = state_dict["model.0.weight"].shape[1]
        nc = state_dict["model.0.weight"].shape[0]

        layers = get_seq_len(state_dict, "model")
        out_nc = state_dict[f"model.{layers-1}.weight"].shape[0]

        if "model.3.weight" in state_dict:
            act_mode = "BR"
            nb = (layers - 3) // 3 + 2
        else:
            act_mode = "R"
            nb = (layers - 3) // 2 + 2

        if in_nc != out_nc:
            mode = "FDnCNN"

        model = DnCNN(
            in_nc=in_nc,
            out_nc=out_nc,
            nc=nc,
            nb=nb,
            act_mode=act_mode,
            mode=mode,
        )

        tags = [f"{nc}nc", f"{nb}nb"]
        if mode == "FDnCNN":
            tags.insert(0, "FDnCNN")
            in_nc -= 1

        def call(model: DnCNN, image: torch.Tensor) -> torch.Tensor:
            if model.mode == "FDnCNN":
                # add noise level map
                _, _, H, W = image.shape  # noqa: N806

                noise_level = 15 / 255  # default from repo
                noise_map = torch.zeros(1, 1, H, W).to(image) + noise_level

                return model(torch.cat([image, noise_map], dim=1))
            else:
                return model(image)

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration",
            tags=tags,
            supports_half=False,  # TODO: verify
            supports_bfloat16=True,
            scale=1,
            input_channels=in_nc,
            output_channels=out_nc,
            size_requirements=SizeRequirements(),
            call_fn=call,
        )


__all__ = ["DnCNNArch", "DnCNN"]
