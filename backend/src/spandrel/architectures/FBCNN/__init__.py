from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import Architecture, ImageModelDescriptor, StateDict
from .__arch.FBCNN import FBCNN


class FBCNNArch(Architecture[FBCNN]):
    def __init__(self) -> None:
        super().__init__(
            id="FBCNN",
            detect=KeyCondition.has_all(
                "m_head.weight",
                "m_down1.0.res.0.weight",
                "m_down2.0.res.0.weight",
                "m_body_encoder.0.res.0.weight",
                "m_tail.weight",
                "qf_pred.0.res.0.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[FBCNN]:
        in_nc = 3
        out_nc = 3
        nc = [64, 128, 256, 512]
        nb = 4
        act_mode = "R"
        downsample_mode = "strideconv"
        upsample_mode = "convtranspose"

        in_nc = state_dict["m_head.weight"].shape[1]
        out_nc = state_dict["m_tail.weight"].shape[0]

        nb = get_seq_len(state_dict, "m_body_encoder")

        nc[0] = state_dict["m_head.weight"].shape[0]
        nc[1] = state_dict["m_down2.0.res.0.weight"].shape[0]
        nc[2] = state_dict["m_down3.0.res.0.weight"].shape[0]
        nc[3] = state_dict["m_body_encoder.0.res.0.weight"].shape[0]

        if f"m_down1.{nb}.weight" in state_dict:
            downsample_mode = "strideconv"
        else:
            # It's either "avgpool" or "maxpool".
            # We cannot detect this from the state dict alone.
            downsample_mode = "avgpool"

        if "m_up3.0.weight" in state_dict:
            upsample_mode = "convtranspose"
        elif "m_up3.0.1.weight" in state_dict:
            upsample_mode = "upconv"
        elif "m_up3.0.0.weight" in state_dict:
            upsample_mode = "pixelshuffle"
        else:
            raise ValueError("Unable to detect upsample mode")

        model = FBCNN(
            in_nc=in_nc,
            out_nc=out_nc,
            nc=nc,
            nb=nb,
            act_mode=act_mode,
            downsample_mode=downsample_mode,
            upsample_mode=upsample_mode,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration",
            tags=[],
            supports_half=True,  # TODO
            supports_bfloat16=True,  # TODO
            scale=1,
            input_channels=in_nc,
            output_channels=out_nc,
            call_fn=lambda model, image: model(image)[0],
        )


__all__ = ["FBCNNArch", "FBCNN"]
