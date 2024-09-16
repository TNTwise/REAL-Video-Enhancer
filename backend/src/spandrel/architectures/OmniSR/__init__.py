import math

from typing_extensions import override

from spandrel.util import KeyCondition, get_scale_and_output_channels, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.OmniSR import OmniSR


class OmniSRArch(Architecture[OmniSR]):
    def __init__(self) -> None:
        super().__init__(
            id="OmniSR",
            detect=KeyCondition.has_all(
                "residual_layer.0.residual_layer.0.layer.0.fn.0.weight",
                "input.weight",
                "up.0.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[OmniSR]:
        # Remove junk from the state dict
        state_dict_keys = set(state_dict.keys())
        for key in state_dict_keys:
            if key.endswith(("total_ops", "total_params")):
                del state_dict[key]

        num_in_ch = 3
        num_out_ch = 3
        num_feat = 64
        block_num = 1
        pe = True
        window_size = 8
        res_num = 1
        up_scale = 4
        bias = True

        num_feat = state_dict["input.weight"].shape[0]
        num_in_ch = state_dict["input.weight"].shape[1]
        bias = "input.bias" in state_dict

        pixelshuffle_shape = state_dict["up.0.weight"].shape[0]
        up_scale, num_out_ch = get_scale_and_output_channels(
            pixelshuffle_shape, num_in_ch
        )

        res_num = get_seq_len(state_dict, "residual_layer")
        block_num = get_seq_len(state_dict, "residual_layer.0.residual_layer") - 1

        rel_pos_bias_key = (
            "residual_layer.0.residual_layer.0.layer.2.fn.rel_pos_bias.weight"
        )
        if rel_pos_bias_key in state_dict:
            pe = True
            # rel_pos_bias_weight = (2 * window_size - 1) ** 2
            rel_pos_bias_weight = state_dict[rel_pos_bias_key].shape[0]
            window_size = int((math.sqrt(rel_pos_bias_weight) + 1) / 2)
        else:
            pe = False

        model = OmniSR(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            num_feat=num_feat,
            block_num=block_num,
            pe=pe,
            window_size=window_size,
            res_num=res_num,
            up_scale=up_scale,
            bias=bias,
        )

        tags = [
            f"{num_feat}nf",
            f"w{window_size}",
            f"{res_num}nr",
        ]

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if up_scale == 1 else "SR",
            tags=tags,
            supports_half=True,  # TODO: Test this
            supports_bfloat16=True,
            scale=up_scale,
            input_channels=num_in_ch,
            output_channels=num_out_ch,
            size_requirements=SizeRequirements(minimum=16),
        )


__all__ = ["OmniSRArch", "OmniSR"]
