from __future__ import annotations

from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.fftformer_arch import FFTformer


class FFTformerArch(Architecture[FFTformer]):
    def __init__(self) -> None:
        super().__init__(
            id="FFTformer",
            detect=KeyCondition.has_all(
                "patch_embed.proj.weight",
                "encoder_level1.0.norm2.body.weight",
                "encoder_level1.0.norm2.body.bias",
                "encoder_level1.0.ffn.fft",
                "encoder_level1.0.ffn.project_in.weight",
                "encoder_level1.0.ffn.dwconv.weight",
                "encoder_level1.0.ffn.project_out.weight",
                "down1_2.body.1.weight",
                "encoder_level2.0.ffn.fft",
                "down2_3.body.1.weight",
                "encoder_level3.0.ffn.fft",
                "decoder_level3.0.attn.to_hidden.weight",
                "decoder_level3.0.attn.norm.body.weight",
                "up3_2.body.1.weight",
                "reduce_chan_level2.weight",
                "decoder_level2.0.attn.to_hidden.weight",
                "up2_1.body.1.weight",
                "decoder_level1.0.attn.to_hidden.weight",
                "refinement.0.norm1.body.weight",
                "refinement.0.attn.to_hidden.weight",
                "refinement.0.ffn.fft",
                "fuse2.att_channel.norm2.body.weight",
                "fuse2.att_channel.ffn.fft",
                "fuse2.conv.weight",
                "fuse1.att_channel.norm2.body.weight",
                "fuse1.att_channel.ffn.fft",
                "fuse1.conv.weight",
                "output.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[FFTformer]:
        inp_channels = 3
        out_channels = 3
        dim = 48
        num_blocks = [6, 6, 12]
        num_refinement_blocks = 4
        ffn_expansion_factor = 3
        bias = False

        inp_channels = state_dict["patch_embed.proj.weight"].shape[1]
        out_channels = state_dict["output.weight"].shape[0]
        dim = state_dict["patch_embed.proj.weight"].shape[0]

        num_blocks[0] = get_seq_len(state_dict, "encoder_level1")
        num_blocks[1] = get_seq_len(state_dict, "encoder_level2")
        num_blocks[2] = get_seq_len(state_dict, "encoder_level3")

        num_refinement_blocks = get_seq_len(state_dict, "refinement")

        # hidden_dim = int(dim * ffn_expansion_factor)
        hidden_dim = state_dict["encoder_level1.0.ffn.project_out.weight"].shape[1]
        ffn_expansion_factor = hidden_dim / dim

        bias = "encoder_level1.0.ffn.project_in.bias" in state_dict

        model = FFTformer(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration",
            tags=[f"{dim}dim"],
            supports_half=False,  # TODO: verify
            supports_bfloat16=True,
            scale=1,
            input_channels=inp_channels,
            output_channels=out_channels,
            size_requirements=SizeRequirements(multiple_of=32),
        )


__all__ = ["FFTformerArch", "FFTformer"]
