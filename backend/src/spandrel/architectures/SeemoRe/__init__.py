import math

from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.seemore_arch import LRSpace, SeemoRe


def _get_lr_space(state_dict: StateDict) -> LRSpace:
    # This is done by detecting the low_dim of the experts and deducing the lr_space
    # from it. This can be done, because the low_dim is determined by the lr_space.
    #
    # The following LR spaces will produce the following low_dim in the first few experts:
    # - linear: 2, 3, 4, 5, ...
    # - double: 2, 4, 6, 8, ...
    # - exp: 2, 4, 8, 16, ...

    # check the 3rd expert first
    low_dim_2_key = "body.0.local_block.block.moe_layer.experts.2.conv_1.weight"
    if low_dim_2_key in state_dict:
        low_dim_2 = state_dict[low_dim_2_key].shape[0]
        if low_dim_2 == 4:
            return "linear"
        elif low_dim_2 == 6:
            return "double"
        elif low_dim_2 == 8:
            return "exp"
        else:
            raise ValueError(f"Unknown low_dim_2: {low_dim_2}")

    # if there is no 3rd expert, check the 2nd expert
    low_dim_1_key = "body.0.local_block.block.moe_layer.experts.1.conv_1.weight"
    if low_dim_1_key in state_dict:
        low_dim_1 = state_dict[low_dim_1_key].shape[0]
        if low_dim_1 == 3:
            return "linear"
        elif low_dim_1 == 4:
            return "double"  # or "exp"
        else:
            raise ValueError(f"Unknown low_dim_1: {low_dim_1}")

    # there's only one expert, so the growth rate doesn't matter
    return "linear"


class SeemoReArch(Architecture[SeemoRe]):
    def __init__(self) -> None:
        super().__init__(
            id="SeemoRe",
            detect=KeyCondition.has_all(
                "conv_1.weight",
                "conv_1.bias",
                "norm.weight",
                "norm.bias",
                "conv_2.weight",
                "conv_2.bias",
                "upsampler.0.weight",
                "upsampler.0.bias",
                "body.0.local_block.norm_1.weight",
                "body.0.local_block.norm_1.bias",
                "body.0.local_block.block.conv_1.0.weight",
                "body.0.local_block.block.conv_1.2.weight",
                "body.0.local_block.block.agg_conv.0.weight",
                "body.0.local_block.block.conv.0.weight",
                "body.0.local_block.block.conv.1.weight",
                "body.0.local_block.block.conv_2.0.conv.0.weight",
                "body.0.local_block.block.conv_2.0.conv.1.weight",
                "body.0.local_block.block.moe_layer.experts.0.conv_1.weight",
                "body.0.local_block.block.moe_layer.experts.0.conv_2.weight",
                "body.0.local_block.block.moe_layer.experts.0.conv_3.weight",
                "body.0.local_block.block.proj.weight",
                "body.0.local_block.norm_2.weight",
                "body.0.local_block.ffn.gate.weight",
                "body.0.global_block.norm_1.weight",
                "body.0.global_block.block.proj.weight",
                "body.0.global_block.block.attn.conv.0.weight",
                "body.0.global_block.ffn.fn_2.0.weight",
                "body.0.global_block.ffn.gate.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[SeemoRe]:
        # default values
        # scale: int = 4
        # in_chans: int = 3
        # num_experts: int = 6
        # num_layers: int = 6
        # embedding_dim: int = 64
        img_range: float = 1.0  # undetectable
        use_shuffle: bool = True  # undetectable
        # global_kernel_size: int = 11
        recursive: int = 2  # undetectable
        lr_space: LRSpace = "linear"
        topk: int = 1  # undetectable

        # detect
        in_chans = state_dict["conv_1.weight"].shape[1]
        embedding_dim = state_dict["conv_1.weight"].shape[0]
        num_layers = get_seq_len(state_dict, "body")

        num_experts = get_seq_len(
            state_dict, "body.0.local_block.block.moe_layer.experts"
        )
        lr_space = _get_lr_space(state_dict)

        scale = math.isqrt(state_dict["upsampler.0.weight"].shape[0] // in_chans)

        global_kernel_size = state_dict[
            "body.0.global_block.block.attn.conv.0.weight"
        ].shape[3]

        extra_tags = []
        if num_layers == 6 and embedding_dim == 36:
            extra_tags = ["Tiny"]
            use_shuffle = True
            topk = 1
            recursive = 2
        elif num_layers == 8 and embedding_dim == 48:
            extra_tags = ["Big"]
            use_shuffle = True
            topk = 1
            recursive = 2
        elif num_layers == 16 and embedding_dim == 48:
            extra_tags = ["Large"]
            use_shuffle = False
            topk = 1
            recursive = 1

        model = SeemoRe(
            scale=scale,
            in_chans=in_chans,
            num_experts=num_experts,
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            img_range=img_range,
            use_shuffle=use_shuffle,
            global_kernel_size=global_kernel_size,
            recursive=recursive,
            lr_space=lr_space,
            topk=topk,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if scale == 1 else "SR",
            tags=[
                *extra_tags,
                f"{embedding_dim}dim",
                f"{num_experts}ne",
                f"{num_layers}nl",
            ],
            supports_half=False,  # TODO: verify
            supports_bfloat16=True,
            scale=scale,
            input_channels=in_chans,
            output_channels=in_chans,
            size_requirements=SizeRequirements(minimum=16),
        )


__all__ = ["SeemoReArch", "SeemoRe"]
