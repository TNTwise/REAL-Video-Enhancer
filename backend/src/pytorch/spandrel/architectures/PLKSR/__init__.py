from __future__ import annotations

import math
from typing import Literal, Sequence, Union

from typing_extensions import override

from ...util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import Architecture, ImageModelDescriptor, StateDict
from .__arch.PLKSR import PLKSR
from .__arch.RealPLKSR import RealPLKSR

_PLKSR = Union[PLKSR, RealPLKSR]

class PLKSRArch(Architecture[_PLKSR]):
    def __init__(self) -> None:
        super().__init__(
            id="PLKSR",
            detect=KeyCondition.has_all(
                "feats.0.weight",
                KeyCondition.has_any(
                    "feats.1.lk.conv.weight",
                    "feats.1.lk.convs.0.weight",
                    "feats.1.lk.mn_conv.weight",
                ),
                "feats.1.refine.weight",
                KeyCondition.has_any(
                    "feats.1.channe_mixer.0.weight",
                    "feats.1.channel_mixer.0.weight",
                ),
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[_PLKSR]:
        dim = 64
        n_blocks = 28
        scale = 4
        kernel_size = 17
        split_ratio = 0.25
        use_ea = True
        supports_half = True

        dim = state_dict["feats.0.weight"].shape[0]

        total_feat_layers = get_seq_len(state_dict, "feats")
        scale = math.isqrt(
            state_dict[f"feats.{total_feat_layers - 1}.weight"].shape[0] // 3
        )

        use_ea = "feats.1.attn.f.0.weight" in state_dict

        if "feats.1.channe_mixer.0.weight" in state_dict:
            # Yes, the normal version has this typo.
            n_blocks = total_feat_layers - 2

            # ccm_type
            mixer_0_shape = state_dict["feats.1.channe_mixer.0.weight"].shape[2]
            mixer_2_shape = state_dict["feats.1.channe_mixer.2.weight"].shape[2]
            if mixer_0_shape == 3 and mixer_2_shape == 1:
                ccm_type = "CCM"
            elif mixer_0_shape == 3 and mixer_2_shape == 3:
                ccm_type = "DCCM"
            elif mixer_0_shape == 1 and mixer_2_shape == 3:
                ccm_type = "ICCM"
            else:
                raise ValueError("Unknown CCM type")
            more_tags = [ccm_type]

            # lk_type
            lk_type: Literal["PLK", "SparsePLK", "RectSparsePLK"] = "PLK"
            use_max_kernel: bool = False
            sparse_kernels: Sequence[int] = [5, 5, 5, 5]
            sparse_dilations: Sequence[int] = [1, 2, 3, 4]
            with_idt: bool = False  # undetectable

            if "feats.1.lk.conv.weight" in state_dict:
                # PLKConv2d
                lk_type = "PLK"
                kernel_size = state_dict["feats.1.lk.conv.weight"].shape[2]
                split_ratio = state_dict["feats.1.lk.conv.weight"].shape[0] / dim
            elif "feats.1.lk.convs.0.weight" in state_dict:
                # SparsePLKConv2d
                lk_type = "SparsePLK"
                split_ratio = state_dict["feats.1.lk.convs.0.weight"].shape[0] / dim
                # Detecting other parameters for SparsePLKConv2d is praticaly impossible.
                # We cannot detect the values of sparse_dilations at all, we only know it has the same length as sparse_kernels.
                # Detecting the values of sparse_kernels is possible, but we don't know its length exactly, because it's `len(sparse_kernels) = len(convs) - (1 if use_max_kernel else 0)`.
                # However, we cannot detect use_max_kernel, because the convolutions it adds when enabled look the same as the other convolutions.
                # So I give up.
            elif "feats.1.lk.mn_conv.weight" in state_dict:
                # RectSparsePLKConv2d
                lk_type = "RectSparsePLK"
                kernel_size = state_dict["feats.1.lk.mn_conv.weight"].shape[2]
                split_ratio = state_dict["feats.1.lk.mn_conv.weight"].shape[0] / dim
            else:
                raise ValueError("Unknown LK type")

            model = PLKSR(
                dim=dim,
                n_blocks=n_blocks,
                upscaling_factor=scale,
                ccm_type=ccm_type,
                kernel_size=kernel_size,
                split_ratio=split_ratio,
                lk_type=lk_type,
                use_max_kernel=use_max_kernel,
                sparse_kernels=sparse_kernels,
                sparse_dilations=sparse_dilations,
                with_idt=with_idt,
                use_ea=use_ea,
            )
        elif "feats.1.channel_mixer.0.weight" in state_dict:
            # and RealPLKSR doesn't. This makes it really convenient to detect.
            more_tags = ["Real"]

            n_blocks = total_feat_layers - 3
            kernel_size = state_dict["feats.1.lk.conv.weight"].shape[2]
            split_ratio = state_dict["feats.1.lk.conv.weight"].shape[0] / dim
            use_layer_norm = "feats.1.layer_norm.bias" in state_dict
            use_dysample = "to_img.init_pos" in state_dict
            if use_dysample:
                more_tags.append("DySample")
            if use_layer_norm:
                more_tags.append("LayerNorm")
            else:
                supports_half = False

            model = RealPLKSR(
                dim=dim,
                upscaling_factor=scale,
                n_blocks=n_blocks,
                kernel_size=kernel_size,
                split_ratio=split_ratio,
                use_ea=use_ea,
                norm_groups=4,  # un-detectable
                dysample=use_dysample,
                layer_norm=use_layer_norm,
            )
        else:
            raise ValueError("Unknown model type")

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if scale == 1 else "SR",
            tags=[f"{dim}dim", f"{n_blocks}nb", f"{kernel_size}ks", *more_tags],
            supports_half=supports_half,
            supports_bfloat16=True,
            scale=scale,
            input_channels=3,
            output_channels=3,
        )


__all__ = ["PLKSRArch", "PLKSR", "RealPLKSR"]
