from __future__ import annotations

from typing import Union

from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.kbnet_l import KBNet_l
from .__arch.kbnet_s import KBNet_s

# KBCNN is essentially 2 similar but different architectures: KBNet_l and KBNet_s.
_KBNet = Union[KBNet_l, KBNet_s]


class KBNetArch(Architecture[_KBNet]):
    def __init__(self) -> None:
        super().__init__(
            id="KBNet",
            detect=KeyCondition.has_any(
                KeyCondition.has_all(
                    # KBNet_s
                    "intro.weight",
                    "encoders.0.0.attgamma",
                    "middle_blks.0.w",
                    "decoders.0.0.attgamma",
                    "ending.weight",
                ),
                KeyCondition.has_all(
                    # KBNet_l
                    "patch_embed.proj.weight",
                    "encoder_level3.0.ffn.project_out.weight",
                    "latent.0.ffn.qkv.weight",
                    "refinement.0.attn.dwconv.0.weight",
                ),
            ),
        )

    def _load_l(self, state_dict: StateDict):
        in_nc = 3
        out_nc = 3
        dim = 48
        num_blocks = [4, 6, 6, 8]
        num_refinement_blocks = 4
        heads = [1, 2, 4, 8]
        ffn_expansion_factor = 1.5
        bias = False

        in_nc = state_dict["patch_embed.proj.weight"].shape[1]
        out_nc = state_dict["output.weight"].shape[0]

        dim = state_dict["patch_embed.proj.weight"].shape[0]

        num_blocks[0] = get_seq_len(state_dict, "encoder_level1")
        num_blocks[1] = get_seq_len(state_dict, "encoder_level2")
        num_blocks[2] = get_seq_len(state_dict, "encoder_level3")
        num_blocks[3] = get_seq_len(state_dict, "latent")

        num_refinement_blocks = get_seq_len(state_dict, "refinement")

        heads[0] = state_dict["encoder_level1.0.ffn.temperature"].shape[0]
        heads[1] = state_dict["encoder_level2.0.ffn.temperature"].shape[0]
        heads[2] = state_dict["encoder_level3.0.ffn.temperature"].shape[0]
        heads[3] = state_dict["latent.0.ffn.temperature"].shape[0]

        bias = "encoder_level1.0.ffn.qkv.bias" in state_dict

        # in code: hidden_features = int(dim * ffn_expansion_factor)
        hidden_features = state_dict["encoder_level1.0.attn.ga1"].shape[1]
        ffn_expansion_factor = hidden_features / dim

        model = KBNet_l(
            inp_channels=in_nc,
            out_channels=out_nc,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration",
            tags=["L", f"{dim}dim"],
            supports_half=False,
            supports_bfloat16=True,
            scale=1,
            input_channels=in_nc,
            output_channels=out_nc,
            size_requirements=SizeRequirements(multiple_of=16),
        )

    def _load_s(self, state_dict: StateDict):
        img_channel = 3
        width = 64
        middle_blk_num = 12
        enc_blk_nums = [2, 2, 4, 8]
        dec_blk_nums = [2, 2, 2, 2]
        lightweight = False
        ffn_scale = 2

        img_channel = state_dict["intro.weight"].shape[1]
        width = state_dict["intro.weight"].shape[0]

        middle_blk_num = get_seq_len(state_dict, "middle_blks")

        enc_count = get_seq_len(state_dict, "encoders")
        enc_blk_nums = [1] * enc_count
        for i in range(enc_count):
            enc_blk_nums[i] = get_seq_len(state_dict, f"encoders.{i}")

        dec_count = get_seq_len(state_dict, "decoders")
        dec_blk_nums = [1] * dec_count
        for i in range(dec_count):
            dec_blk_nums[i] = get_seq_len(state_dict, f"decoders.{i}")

        # in code: ffn_ch = int(c * ffn_scale)
        temp_c = state_dict["middle_blks.0.conv4.weight"].shape[1]
        temp_ffn_ch = state_dict["middle_blks.0.conv4.weight"].shape[0]
        ffn_scale = temp_ffn_ch / temp_c

        # kernel size is 3 for lightweight and 5 otherwise
        kernel_size = state_dict["encoders.0.0.conv11.1.weight"].shape[2]
        lightweight = kernel_size == 3

        model = KBNet_s(
            img_channel=img_channel,
            width=width,
            middle_blk_num=middle_blk_num,
            enc_blk_nums=enc_blk_nums,
            dec_blk_nums=dec_blk_nums,
            lightweight=lightweight,
            ffn_scale=ffn_scale,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration",
            tags=[
                "S",
                f"{width}w",
                *(["lightweight"] if lightweight else []),
            ],
            supports_half=False,
            supports_bfloat16=True,
            scale=1,
            input_channels=img_channel,
            output_channels=img_channel,
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[_KBNet]:
        if "patch_embed.proj.weight" in state_dict:
            return self._load_l(state_dict)
        else:
            return self._load_s(state_dict)


__all__ = ["KBNetArch", "KBNet_s", "KBNet_l"]
