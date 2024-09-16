from __future__ import annotations

import torch
from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    ModelTiling,
    SizeRequirements,
    StateDict,
)
from .__arch.retinexformer_arch import RetinexFormer


def _call_fn(model: RetinexFormer, t: torch.Tensor) -> torch.Tensor:
    h, w = t.shape[-2:]

    if h < 3000 and w < 3000:
        return model(t)

    # this uses interlacing to split the image into 2 smaller parts
    restored = torch.zeros_like(t)
    restored[:, :, :, 1::2] = model(t[:, :, :, 1::2])
    restored[:, :, :, 0::2] = model(t[:, :, :, 0::2])
    return restored


class RetinexFormerArch(Architecture[RetinexFormer]):
    def __init__(self) -> None:
        super().__init__(
            id="RetinexFormer",
            detect=KeyCondition.has_all(
                "body.0.estimator.conv1.weight",
                "body.0.estimator.conv1.bias",
                "body.0.estimator.depth_conv.weight",
                "body.0.estimator.depth_conv.bias",
                "body.0.estimator.conv2.weight",
                "body.0.estimator.conv2.bias",
                "body.0.denoiser.embedding.weight",
                "body.0.denoiser.mapping.weight",
                "body.0.denoiser.encoder_layers.0.0.blocks.0.0.rescale",
                "body.0.denoiser.encoder_layers.0.0.blocks.0.0.to_q.weight",
                "body.0.denoiser.encoder_layers.0.0.blocks.0.0.to_v.weight",
                "body.0.denoiser.encoder_layers.0.0.blocks.0.0.to_k.weight",
                "body.0.denoiser.encoder_layers.0.0.blocks.0.0.proj.weight",
                "body.0.denoiser.encoder_layers.0.0.blocks.0.0.pos_emb.0.weight",
                "body.0.denoiser.encoder_layers.0.0.blocks.0.1.fn.net.0.weight",
                "body.0.denoiser.encoder_layers.0.0.blocks.0.1.norm.weight",
                "body.0.denoiser.encoder_layers.0.1.weight",
                "body.0.denoiser.encoder_layers.0.2.weight",
                "body.0.denoiser.bottleneck.blocks.0.0.rescale",
                "body.0.denoiser.decoder_layers.0.0.weight",
                "body.0.denoiser.decoder_layers.0.1.weight",
                "body.0.denoiser.decoder_layers.0.2.blocks.0.0.rescale",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[RetinexFormer]:
        in_channels = 3
        out_channels = 3
        n_feat = 40
        stage = 3
        num_blocks = [1, 1, 1]

        stage = get_seq_len(state_dict, "body")

        n_feat = state_dict["body.0.denoiser.embedding.weight"].shape[0]
        in_channels = state_dict["body.0.denoiser.embedding.weight"].shape[1]
        out_channels = state_dict["body.0.denoiser.mapping.weight"].shape[0]

        num_blocks = [
            get_seq_len(state_dict, "body.0.denoiser.encoder_layers.0.0.blocks"),
            get_seq_len(state_dict, "body.0.denoiser.encoder_layers.1.0.blocks"),
            get_seq_len(state_dict, "body.0.denoiser.bottleneck.blocks"),
        ]

        model = RetinexFormer(
            in_channels=in_channels,
            out_channels=out_channels,
            n_feat=n_feat,
            stage=stage,
            num_blocks=num_blocks,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration",
            tags=[
                f"{n_feat}nf",
                f"{stage}s",
                f"{num_blocks[0]}x{num_blocks[1]}x{num_blocks[2]}b",
            ],
            supports_half=False,  # TODO: verify
            supports_bfloat16=True,
            scale=1,
            input_channels=in_channels,
            output_channels=out_channels,
            size_requirements=SizeRequirements(multiple_of=8),
            tiling=ModelTiling.DISCOURAGED,
            call_fn=_call_fn,
        )


__all__ = ["RetinexFormerArch", "RetinexFormer"]
