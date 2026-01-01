import math

from typing_extensions import override
from typing_extensions import override

from ...util import KeyCondition, get_scale_and_output_channels, get_seq_len

from ...__helpers.model_descriptor import Architecture, ImageModelDescriptor, StateDict

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.artcnn_arch import ArtCNN


class ArtCNNArch(Architecture[ArtCNN]):
    def __init__(
        self,
    ) -> None:
        super().__init__(
            id="ArtCNN",
            name="ArtCNN",
            detect=KeyCondition.has_all(
                "depth_to_space.upscale.0.bias",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[ArtCNN]:
        in_ch = 3
        scale = 2
        filters = 96
        n_block = 16
        kernel_size = 3
        model = ArtCNN(
            in_ch=in_ch,
            scale=scale,
            filters=filters,
            n_block=n_block,
            kernel_size=kernel_size,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if scale == 1 else "SR",
            tags=[f"{filters}nf", f"{n_block}nc"],
            supports_half=True,
            supports_bfloat16=True,
            scale=scale,
            input_channels=in_ch,
            output_channels=in_ch,
        )



__all__ = ["ArtCNNArch", "ArtCNN"]
