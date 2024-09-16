from typing_extensions import override

from spandrel.util import KeyCondition, get_scale_and_output_channels

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.dctlsa import DCTLSA


class DCTLSAArch(Architecture[DCTLSA]):
    def __init__(self) -> None:
        super().__init__(
            id="DCTLSA",
            detect=KeyCondition.has_all(
                "fea_conv.weight",
                "B1.body.0.transformer_body.0.blocks.0.attn.qkv.weight",
                "B1.body.0.transformer_body.0.blocks.0.attn.local.pointwise_prenorm_1.weight",
                "B1.body.1.transformer_body.0.blocks.0.attn.qkv.weight",
                "B1.body.1.transformer_body.0.blocks.0.attn.local.pointwise_prenorm_1.weight",
                "B6.body.0.transformer_body.0.blocks.0.attn.qkv.weight",
                "B6.body.0.transformer_body.0.blocks.0.attn.local.pointwise_prenorm_1.weight",
                "B6.body.1.transformer_body.0.blocks.0.attn.qkv.weight",
                "B6.body.1.transformer_body.0.blocks.0.attn.local.pointwise_prenorm_1.weight",
                "c.0.weight",
                "c1.0.weight",
                "c2.0.weight",
                "c3.0.weight",
                "c4.0.weight",
                "c5.0.weight",
                "LR_conv.weight",
                "upsampler.0.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[DCTLSA]:
        # defaults
        in_nc = 3
        nf = 55
        num_modules = 6
        out_nc = 3
        upscale = 4
        num_head = 5  # cannot be deduced from state dict

        in_nc = state_dict["fea_conv.weight"].shape[1]
        nf = state_dict["fea_conv.weight"].shape[0]
        num_modules = state_dict["c.0.weight"].shape[1] // nf

        # good old pixelshuffle
        x = state_dict["upsampler.0.weight"].shape[0]
        upscale, out_nc = get_scale_and_output_channels(x, in_nc)

        model = DCTLSA(
            in_nc=in_nc,
            nf=nf,
            num_modules=num_modules,
            out_nc=out_nc,
            upscale=upscale,
            num_head=num_head,
        )

        tags = [
            f"{nf}nf",
            f"{num_modules}nm",
            f"{num_head}nh",
        ]

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if upscale == 1 else "SR",
            tags=tags,
            supports_half=False,  # TODO: test
            supports_bfloat16=True,
            scale=upscale,
            input_channels=in_nc,
            output_channels=out_nc,
            size_requirements=SizeRequirements(minimum=16),
        )


__all__ = ["DCTLSAArch", "DCTLSA"]
