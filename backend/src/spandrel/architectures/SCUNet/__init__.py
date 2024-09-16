from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    ModelTiling,
    SizeRequirements,
    StateDict,
)
from .__arch.SCUNet import SCUNet


class SCUNetArch(Architecture[SCUNet]):
    def __init__(self) -> None:
        super().__init__(
            id="SCUNet",
            detect=KeyCondition.has_all(
                "m_head.0.weight",
                "m_tail.0.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[SCUNet]:
        in_nc = 3
        config = [4, 4, 4, 4, 4, 4, 4]
        dim = 64
        drop_path_rate = 0.0
        input_resolution = 256

        dim = state_dict["m_head.0.weight"].shape[0]
        in_nc = state_dict["m_head.0.weight"].shape[1]

        config[0] = get_seq_len(state_dict, "m_down1") - 1
        config[1] = get_seq_len(state_dict, "m_down2") - 1
        config[2] = get_seq_len(state_dict, "m_down3") - 1
        config[3] = get_seq_len(state_dict, "m_body")
        config[4] = get_seq_len(state_dict, "m_up3") - 1
        config[5] = get_seq_len(state_dict, "m_up2") - 1
        config[6] = get_seq_len(state_dict, "m_up1") - 1

        model = SCUNet(
            in_nc=in_nc,
            config=config,
            dim=dim,
            drop_path_rate=drop_path_rate,
            input_resolution=input_resolution,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration",
            tags=[],
            supports_half=True,
            supports_bfloat16=True,
            scale=1,
            input_channels=in_nc,
            output_channels=in_nc,
            size_requirements=SizeRequirements(minimum=40),
            tiling=ModelTiling.DISCOURAGED,
        )


__all__ = ["SCUNetArch", "SCUNet"]
