import math

from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.safmn import SAFMN


class SAFMNArch(Architecture[SAFMN]):
    def __init__(self) -> None:
        super().__init__(
            id="SAFMN",
            detect=KeyCondition.has_all(
                "to_feat.weight",
                "feats.0.norm1.weight",
                "feats.0.norm2.weight",
                "feats.0.safm.mfr.0.weight",
                "feats.0.safm.mfr.3.weight",
                "feats.0.safm.aggr.weight",
                "feats.0.ccm.ccm.0.weight",
                "feats.0.ccm.ccm.2.weight",
                "to_img.0.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[SAFMN]:
        dim: int
        n_blocks: int = 8
        ffn_scale: float = 2.0
        upscaling_factor: int = 4

        dim = state_dict["to_feat.weight"].shape[0]
        n_blocks = get_seq_len(state_dict, "feats")

        # hidden_dim = int(dim * ffn_scale)
        hidden_dim = state_dict["feats.0.ccm.ccm.0.weight"].shape[0]
        ffn_scale = hidden_dim / dim

        # 3 * upscaling_factor**2
        upscaling_factor = int(math.sqrt(state_dict["to_img.0.weight"].shape[0] / 3))

        model = SAFMN(
            dim=dim,
            n_blocks=n_blocks,
            ffn_scale=ffn_scale,
            upscaling_factor=upscaling_factor,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if upscaling_factor == 1 else "SR",
            tags=[f"{dim}dim", f"{n_blocks}nb"],
            supports_half=False,  # TODO: verify
            supports_bfloat16=True,
            scale=upscaling_factor,
            input_channels=3,  # hard-coded in the arch
            output_channels=3,  # hard-coded in the arch
            size_requirements=SizeRequirements(multiple_of=8),
        )


__all__ = ["SAFMNArch", "SAFMN"]
