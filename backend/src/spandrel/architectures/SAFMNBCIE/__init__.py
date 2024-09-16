import math

from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.safmn_bcie import SAFMN_BCIE as SAFMNBCIE


class SAFMNBCIEArch(Architecture[SAFMNBCIE]):
    def __init__(self) -> None:
        super().__init__(
            id="SAFMNBCIE",
            name="SAFMN BCIE",
            detect=KeyCondition.has_all(
                "to_feat.1.weight",
                "to_feat.1.bias",
                "feats.0.layers.0.norm1.weight",
                "feats.0.layers.0.norm2.weight",
                "feats.0.layers.0.safm.mfr.0.weight",
                "feats.0.layers.0.safm.mfr.3.weight",
                "feats.0.layers.0.ccm.ccm.0.weight",
                "feats.0.layers.0.ccm.ccm.2.weight",
                "feats.0.conv.weight",
                "feats.0.conv.bias",
                "to_img.0.weight",
                "to_img.0.bias",
                "to_img.2.weight",
                "to_img.2.bias",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[SAFMNBCIE]:
        dim: int
        n_blocks: int = 6
        num_layers: int = 6
        ffn_scale: float = 2.0
        upscaling_factor: int = 2

        dim = state_dict["to_feat.1.weight"].shape[0]
        # 3 * upscaling_factor**2
        upscaling_factor = math.isqrt(state_dict["to_feat.1.weight"].shape[1] // 3)

        n_blocks = get_seq_len(state_dict, "feats")
        num_layers = get_seq_len(state_dict, "feats.0.layers")

        # hidden_dim = int(dim * ffn_scale)
        hidden_dim = state_dict["feats.0.layers.0.ccm.ccm.0.weight"].shape[0]
        ffn_scale = hidden_dim / dim

        model = SAFMNBCIE(
            dim=dim,
            n_blocks=n_blocks,
            num_layers=num_layers,
            ffn_scale=ffn_scale,
            upscaling_factor=upscaling_factor,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration",
            tags=[
                f"{dim}dim",
                f"{num_layers}nl",
                f"{n_blocks}nb",
                f"{upscaling_factor}uf",
            ],
            supports_half=False,  # TODO: verify
            supports_bfloat16=True,
            scale=1,
            input_channels=3,  # hard-coded in the arch
            output_channels=3,  # hard-coded in the arch
            size_requirements=SizeRequirements(multiple_of=16),
        )


__all__ = ["SAFMNBCIEArch", "SAFMNBCIE"]
