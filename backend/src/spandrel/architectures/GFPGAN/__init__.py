from typing_extensions import override

from spandrel.util import KeyCondition

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.gfpganv1_clean_arch import GFPGANv1Clean as GFPGAN


class GFPGANArch(Architecture[GFPGAN]):
    def __init__(self) -> None:
        super().__init__(
            id="GFPGAN",
            detect=KeyCondition.has_all(
                "toRGB.0.weight",
                "stylegan_decoder.style_mlp.1.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[GFPGAN]:
        out_size = 512
        num_style_feat = 512
        channel_multiplier = 2
        decoder_load_path = None
        fix_decoder = False
        num_mlp = 8
        input_is_latent = True
        different_w = True
        narrow = 1
        sft_half = True

        model = GFPGAN(
            out_size=out_size,
            num_style_feat=num_style_feat,
            channel_multiplier=channel_multiplier,
            decoder_load_path=decoder_load_path,
            fix_decoder=fix_decoder,
            num_mlp=num_mlp,
            input_is_latent=input_is_latent,
            different_w=different_w,
            narrow=narrow,
            sft_half=sft_half,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="FaceSR",
            tags=[],
            supports_half=False,
            supports_bfloat16=True,
            scale=1,
            input_channels=3,
            output_channels=3,
            size_requirements=SizeRequirements(minimum=512),
            call_fn=lambda model, image: model(image)[0],
        )


__all__ = ["GFPGANArch", "GFPGAN"]
