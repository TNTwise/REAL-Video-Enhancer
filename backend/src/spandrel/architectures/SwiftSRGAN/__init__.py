from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import Architecture, ImageModelDescriptor, StateDict
from .__arch.SwiftSRGAN import Generator as SwiftSRGAN


class SwiftSRGANArch(Architecture[SwiftSRGAN]):
    def __init__(self) -> None:
        super().__init__(
            id="SwiftSRGAN",
            name="Swift-SRGAN",
            detect=KeyCondition.has_all(
                "initial.cnn.depthwise.weight",
                "final_conv.pointwise.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[SwiftSRGAN]:
        in_channels: int = 3
        num_channels: int = 64
        num_blocks: int = 16
        upscale_factor: int = 4

        in_channels = state_dict["initial.cnn.depthwise.weight"].shape[0]
        num_channels = state_dict["initial.cnn.pointwise.weight"].shape[0]
        num_blocks = get_seq_len(state_dict, "residual")
        upscale_factor = 2 ** get_seq_len(state_dict, "upsampler")

        model = SwiftSRGAN(
            in_channels=in_channels,
            num_channels=num_channels,
            num_blocks=num_blocks,
            upscale_factor=upscale_factor,
        )
        tags = [
            f"{num_channels}nf",
            f"{num_blocks}nb",
        ]

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if upscale_factor == 1 else "SR",
            tags=tags,
            supports_half=True,
            supports_bfloat16=True,
            scale=upscale_factor,
            input_channels=in_channels,
            output_channels=in_channels,
        )


__all__ = ["SwiftSRGANArch", "SwiftSRGAN"]
