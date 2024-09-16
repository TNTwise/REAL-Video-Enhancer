from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    MaskedImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.LaMa import LaMa


class LaMaArch(Architecture[LaMa]):
    def __init__(self) -> None:
        super().__init__(
            id="LaMa",
            detect=KeyCondition.has_any(
                "model.model.1.bn_l.running_mean",
                "generator.model.1.bn_l.running_mean",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> MaskedImageModelDescriptor[LaMa]:
        state_dict = {
            k.replace("generator.model", "model.model"): v
            for k, v in state_dict.items()
        }

        in_nc = 4
        out_nc = 3

        in_nc = state_dict["model.model.1.ffc.convl2l.weight"].shape[1]

        seq_len = get_seq_len(state_dict, "model.model")
        out_nc = state_dict[f"model.model.{seq_len - 1}.weight"].shape[0]

        model = LaMa(
            in_nc=in_nc,
            out_nc=out_nc,
        )

        return MaskedImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Inpainting",
            tags=[],
            supports_half=False,
            supports_bfloat16=True,
            input_channels=in_nc - 1,
            output_channels=out_nc,
            size_requirements=SizeRequirements(minimum=16, multiple_of=8),
        )


__all__ = ["LaMaArch", "LaMa"]
