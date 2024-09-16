import torch
from torchvision.transforms.functional import normalize as tv_normalize
from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.restoreformer_arch import RestoreFormer


class RestoreFormerArch(Architecture[RestoreFormer]):
    def __init__(self) -> None:
        super().__init__(
            id="RestoreFormer",
            detect=KeyCondition.has_all(
                "quantize.embedding.weight",
                "encoder.conv_in.weight",
                "quant_conv.weight",
                "encoder.down.0.block.0.norm1.weight",
                "encoder.conv_out.weight",
                "decoder.conv_out.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[RestoreFormer]:
        n_embed = 1024
        embed_dim = 256
        ch = 64
        out_ch = 3
        ch_mult = (1, 2, 2, 4, 4, 8)
        num_res_blocks = 2
        attn_resolutions = (16,)
        dropout = 0.0
        in_channels = 3
        resolution = 512
        z_channels = 256
        double_z = False
        enable_mid = True
        head_size = 8  # cannot be deduced from the shape of tensors in state_dict

        n_embed = state_dict["quantize.embedding.weight"].shape[0]
        embed_dim = state_dict["quantize.embedding.weight"].shape[1]
        z_channels = state_dict["quant_conv.weight"].shape[1]
        double_z = state_dict["encoder.conv_out.weight"].shape[0] == 2 * z_channels

        enable_mid = "encoder.mid.block_1.norm1.weight" in state_dict

        ch = state_dict["encoder.conv_in.weight"].shape[0]
        in_channels = state_dict["encoder.conv_in.weight"].shape[1]
        out_ch = state_dict["decoder.conv_out.weight"].shape[0]

        num_res_blocks = get_seq_len(state_dict, "encoder.down.0.block")

        ch_mult_len = get_seq_len(state_dict, "encoder.down")
        ch_mult_list = [1] * ch_mult_len
        for i in range(ch_mult_len):
            ch_mult_list[i] = (
                state_dict[f"encoder.down.{i}.block.0.conv2.weight"].shape[0] // ch
            )
        ch_mult = tuple(ch_mult_list)

        model = RestoreFormer(
            n_embed=n_embed,
            embed_dim=embed_dim,
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            in_channels=in_channels,
            resolution=resolution,
            z_channels=z_channels,
            double_z=double_z,
            enable_mid=enable_mid,
            head_size=head_size,
        )

        def call(model: RestoreFormer, x: torch.Tensor) -> torch.Tensor:
            x = tv_normalize(x, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            result = model(x)[0]
            return (result + 1) / 2

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="FaceSR",
            tags=[],
            supports_half=False,
            supports_bfloat16=True,
            scale=1,
            input_channels=in_channels,
            output_channels=out_ch,
            size_requirements=SizeRequirements(multiple_of=32),
            call_fn=call,
        )


__all__ = ["RestoreFormerArch", "RestoreFormer"]
