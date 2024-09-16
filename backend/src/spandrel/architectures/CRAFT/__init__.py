import math

from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.CRAFT import CRAFT


class CRAFTArch(Architecture[CRAFT]):
    def __init__(
        self,
    ) -> None:
        super().__init__(
            id="CRAFT",
            detect=KeyCondition.has_all(
                "conv_first.weight",
                "layers.0.residual_group.hf_blocks.0.attn.temperature",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[CRAFT]:
        # default values
        in_chans = 3
        window_size = 16  # cannot be deduced from state dict
        embed_dim = 48
        depths = [2, 2, 2, 2]
        num_heads = [6, 6, 6, 6]
        split_size_0 = 4
        split_size_1 = 16
        mlp_ratio = 2.0
        qkv_bias = True
        qk_scale = None  # cannot be deduced from state dict
        upscale = 4
        img_range = 1.0
        resi_connection = "1conv"

        # detect parameters
        in_chans = state_dict["conv_first.weight"].shape[1]
        embed_dim = state_dict["conv_first.weight"].shape[0]

        num_layers = get_seq_len(state_dict, "layers")
        depths = [2] * num_layers
        num_heads = [6] * num_layers
        for i in range(num_layers):
            depths[i] = get_seq_len(state_dict, f"layers.{i}.residual_group.hf_blocks")
            num_heads[i] = state_dict[
                f"layers.{i}.residual_group.hf_blocks.0.attn.temperature"
            ].shape[0]

        upscale = int(math.sqrt(state_dict["upsample.0.bias"].shape[0] / in_chans))

        if "conv_after_body.weight" in state_dict:
            resi_connection = "1conv"
        else:
            resi_connection = "identity"

        qkv_bias = "layers.0.residual_group.srwa_blocks.0.qkv.bias" in state_dict

        mlp_hidden_dim = state_dict[
            "layers.0.residual_group.srwa_blocks.0.mlp.fc1.bias"
        ].shape[0]
        mlp_ratio = float(mlp_hidden_dim / embed_dim)

        # Now split_size_0 and split_size_1:
        # What we know:
        #   a = s0 * s1
        #   b = (2*s0-1) * (2*s1-1)
        a = state_dict["relative_position_index_h"].shape[0]
        b = state_dict["biases_v"].shape[0]
        # Let's rearrange:
        #   s0 = a / s1
        #   b = (2*a/s1-1) * (2*s1-1)
        # Solve for s1 (wolfram alpha):
        #   s1 = 1/4 (-sqrt(16 a^2 - 8 a (b + 1) + (b - 1)^2) + 4 a - b + 1)
        s1 = int(
            0.25
            * (-math.sqrt(16 * a**2 - 8 * a * (b + 1) + (b - 1) ** 2) + 4 * a - b + 1)
        )
        s0 = a // s1
        if s0 * s1 != a:
            raise ValueError("Could not find valid split_size_0 and split_size_1")
        # since we don't know which is which, we'll just assume split_size_0 <= split_size_1
        split_size_0 = min(s0, s1)
        split_size_1 = max(s0, s1)

        model = CRAFT(
            in_chans=in_chans,
            window_size=window_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            split_size_0=split_size_0,
            split_size_1=split_size_1,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            upscale=upscale,
            img_range=img_range,
            resi_connection=resi_connection,
        )

        tags = [
            f"{split_size_0}x{split_size_1}",
            f"{embed_dim}dim",
            f"{resi_connection}",
        ]

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if upscale == 1 else "SR",
            tags=tags,
            supports_half=True,  # TODO: Not thoroughly tested
            supports_bfloat16=True,
            scale=upscale,
            input_channels=in_chans,
            output_channels=in_chans,
            size_requirements=SizeRequirements(minimum=16, multiple_of=16),
        )


__all__ = ["CRAFTArch", "CRAFT"]
