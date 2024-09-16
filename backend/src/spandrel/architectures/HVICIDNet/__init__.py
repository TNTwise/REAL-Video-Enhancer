from typing_extensions import override

from spandrel.util import KeyCondition

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    ModelTiling,
    SizeRequirements,
    StateDict,
)
from .__arch.cidnet import CIDNet as HVICIDNet


class HVICIDNetArch(Architecture[HVICIDNet]):
    def __init__(self) -> None:
        super().__init__(
            id="HVICIDNet",
            name="HVI-CIDNet",
            detect=KeyCondition.has_all(
                "HVE_block0.1.weight",
                "HVE_block1.prelu.weight",
                "HVE_block1.down.0.weight",
                "HVE_block3.down.0.weight",
                "HVD_block3.prelu.weight",
                "HVD_block3.up_scale.0.weight",
                "HVD_block3.up.weight",
                "HVD_block1.up.weight",
                "HVD_block0.1.weight",
                "IE_block0.1.weight",
                "IE_block1.prelu.weight",
                "IE_block1.down.0.weight",
                "ID_block1.up.weight",
                "ID_block0.1.weight",
                "HV_LCA1.gdfn.project_in.weight",
                "HV_LCA1.gdfn.dwconv.weight",
                "HV_LCA1.gdfn.dwconv1.weight",
                "HV_LCA1.gdfn.dwconv2.weight",
                "HV_LCA1.gdfn.project_out.weight",
                "HV_LCA1.norm.weight",
                "HV_LCA1.ffn.temperature",
                "HV_LCA1.ffn.q.weight",
                "HV_LCA1.ffn.q_dwconv.weight",
                "HV_LCA1.ffn.project_out.weight",
                "HV_LCA2.gdfn.project_in.weight",
                "HV_LCA6.gdfn.project_in.weight",
                "I_LCA1.gdfn.project_in.weight",
                "I_LCA6.ffn.project_out.weight",
                "trans.density_k",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[HVICIDNet]:
        channels = [36, 36, 72, 144]
        heads = [1, 2, 4, 8]
        norm = False

        channels = [
            state_dict["HVE_block1.down.0.weight"].shape[1],
            state_dict["HVE_block1.down.0.weight"].shape[0],
            state_dict["HVE_block2.down.0.weight"].shape[0],
            state_dict["HVE_block3.down.0.weight"].shape[0],
        ]

        heads = [
            1,  # unused
            state_dict["HV_LCA1.ffn.temperature"].shape[0],
            state_dict["HV_LCA2.ffn.temperature"].shape[0],
            state_dict["HV_LCA3.ffn.temperature"].shape[0],
        ]

        norm = "HVE_block1.norm.weight" in state_dict

        model = HVICIDNet(
            channels=channels,
            heads=heads,
            norm=norm,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration",
            tags=[],
            supports_half=False,  # TODO: verify
            supports_bfloat16=True,
            scale=1,
            input_channels=3,  # hard-coded
            output_channels=3,  # hard-coded
            size_requirements=SizeRequirements(multiple_of=8),
            tiling=ModelTiling.DISCOURAGED,
        )


__all__ = ["HVICIDNetArch", "HVICIDNet"]
