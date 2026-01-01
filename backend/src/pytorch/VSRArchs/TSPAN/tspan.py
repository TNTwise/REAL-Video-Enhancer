# Final TSPAN Architecture
# Based on SPAN: https://github.com/hongyuanyu/SPAN/blob/main/basicsr/archs/span_arch.py
# Uses elements from TSCUNet: https://github.com/Kim2091/TSCUNet/blob/main/models/network_tscunet.py

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


def _make_pair(value: Any) -> Any:
    if isinstance(value, int):
        return (value, value)
    return value


def conv_layer(
    in_channels: int, out_channels: int, kernel_size: int, bias: bool = True
) -> nn.Conv2d:
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size_t: tuple[int, int] = _make_pair(kernel_size)
    padding = (int((kernel_size_t[0] - 1) / 2), int((kernel_size_t[1] - 1) / 2))
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def sequential(*args: nn.Module) -> nn.Module:
    """
    Modules will be added to the a Sequential Container in the order they
    are passed.
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        else:
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(
    in_channels: int, out_channels: int, upscale_factor: int = 2, kernel_size: int = 3
) -> nn.Module:
    """
    Upsample features according to `upscale_factor`.
    """
    conv = conv_layer(in_channels, out_channels * (upscale_factor**2), kernel_size)

    # Apply ICNR initialization to prevent checkerboard artifacts
    icnr_init(conv, upscale_factor)

    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


def icnr_init(
    conv: nn.Conv2d,
    upscale_factor: int,
    init_fn: Callable[[Tensor], Tensor] = nn.init.kaiming_normal_,
) -> None:
    """
    ICNR initialization for PixelShuffle layers to prevent checkerboard artifacts.

    Args:
        conv: The convolution layer before PixelShuffle
        upscale_factor: The upscale factor used in PixelShuffle
        init_fn: Initialization function for the base kernel
    """
    # Get output channels and calculate sub-kernel size
    out_channels = conv.out_channels
    sub_kernel_channels = out_channels // (upscale_factor**2)

    # Create a temporary smaller kernel
    sub_kernel = torch.zeros(
        [
            sub_kernel_channels,
            conv.in_channels,
            conv.kernel_size[0],
            conv.kernel_size[1],
        ]
    )

    # Initialize the sub-kernel
    init_fn(sub_kernel)

    # Replicate the sub-kernel for each sub-pixel
    kernel = sub_kernel.repeat_interleave(upscale_factor**2, dim=0)

    # Set the conv weights
    conv.weight.data.copy_(kernel)

    # Initialize bias to zero if it exists
    if conv.bias is not None:
        conv.bias.data.zero_()


class Conv3XC(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        gain1: int = 1,
        s: int = 1,
        bias: Literal[True] = True,
    ) -> None:
        super().__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s
        gain = gain1

        self.sk = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=1,
            padding=0,
            stride=s,
            bias=bias,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=c_in,
                out_channels=c_in * gain,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_in * gain,
                out_channels=c_out * gain,
                kernel_size=3,
                stride=s,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_out * gain,
                out_channels=c_out,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
        )

        self.eval_conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=3,
            padding=1,
            stride=s,
            bias=bias,
        )

        self.eval_conv.weight.requires_grad = False
        if self.eval_conv.bias is not None:
            self.eval_conv.bias.requires_grad = False
        self.update_params()

    def update_params(self) -> None:
        w1 = self.conv[0].weight.data.clone().detach()
        b1 = self.conv[0].bias.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        b2 = self.conv[1].bias.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        b3 = self.conv[2].bias.data.clone().detach()

        w = (
            F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        self.weight_concat = (
            F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()
        target_kernel_size = 3

        h_pixels_to_pad = (target_kernel_size - 1) // 2
        w_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(
            sk_w, [h_pixels_to_pad, h_pixels_to_pad, w_pixels_to_pad, w_pixels_to_pad]
        )

        self.weight_concat = self.weight_concat + sk_w
        self.bias_concat = self.bias_concat + sk_b

        self.eval_conv.weight.data = self.weight_concat.contiguous()
        self.eval_conv.bias.data = self.bias_concat.contiguous()

    def train(self, mode: bool = True) -> Conv3XC:
        super().train(mode)
        # If inference, update params
        if not mode:
            self.update_params()
        return self

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            out = self.eval_conv(x)
        return out


class SPAB(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int | None = None,
        out_channels: int | None = None,
    ) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=2, s=1)
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=2, s=1)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=2, s=1)

        # Add layer normalization
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=mid_channels)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=mid_channels)
        self.norm3 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        self.act1 = torch.nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out1 = self.c1_r(x)
        out1 = self.norm1(out1)
        out1_act = self.act1(out1)

        out2 = self.c2_r(out1_act)
        out2 = self.norm2(out2)
        out2_act = self.act1(out2)

        out3 = self.c3_r(out2_act)
        out3 = self.norm3(out3)

        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att

        return out, out1, sim_att


class TemporalSPAN(nn.Module):
    """
    Temporal version of SPAN using early fusion for video processing.

    This architecture is designed for TensorRT compatibility. It accepts inputs
    in the format (B, T, C, H, W), where T is the number of frames.
    It reshapes the input to (B, T*C, H, W) and processes it through the
    2D convolutional backbone.
    """

    hyperparameters = {}  # noqa: RUF012

    def __init__(
        self,
        *,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_frames: int = 5,
        feature_channels: int = 48,
        upscale: int = 4,
        bias: bool = True,
        history_channels: int = 12,
    ) -> None:
        super().__init__()

        self.num_frames = num_frames
        self.in_channels = num_in_ch
        self.out_channels = num_out_ch
        self.history_channels = history_channels

        # Calculate center frame index
        self.center_idx = num_frames // 2

        # Separate feature extraction for center frame (full dimension)
        self.center_conv = Conv3XC(self.in_channels, feature_channels, gain1=2, s=1)

        # Feature extraction for history frames (reduced dimension)
        # Each history frame gets its own conv to extract features independently
        self.history_convs = nn.ModuleList(
            [
                Conv3XC(self.in_channels, history_channels, gain1=2, s=1)
                for _ in range(num_frames - 1)
            ]
        )

        # Fusion layer to merge all features
        # Total channels: feature_channels (center) + history_channels * (num_frames - 1)
        total_channels = feature_channels + history_channels * (num_frames - 1)
        self.fusion_conv = nn.Sequential(
            Conv3XC(total_channels, feature_channels, gain1=2, s=1),
            Conv3XC(feature_channels, feature_channels, gain1=2, s=1),
        )

        # The rest of the architecture remains the same as the spatial version
        self.block_1 = SPAB(feature_channels)
        self.block_2 = SPAB(feature_channels)
        self.block_3 = SPAB(feature_channels)
        self.block_4 = SPAB(feature_channels)
        self.block_5 = SPAB(feature_channels)
        self.block_6 = SPAB(feature_channels)

        self.conv_cat = conv_layer(
            feature_channels * 4, feature_channels, kernel_size=1, bias=True
        )
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain1=2, s=1)

        self.upsampler = pixelshuffle_block(
            feature_channels, self.out_channels, upscale_factor=upscale
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Processes a sequence of frames.
        Args:
            x: Input tensor with shape (B, T, C, H, W)
        Returns:
            Output tensor with shape (B, C, H_out, W_out)
        """
        # Get input dimensions
        _b, t, _c, _h, _w = x.shape

        # Verify that the number of frames in the input tensor matches the model's configuration
        if t != self.num_frames:
            raise ValueError(
                f"Expected input with {self.num_frames} frames, but received {t} frames."
            )

        # Extract features separately for each frame
        features = []
        history_idx = 0

        for i in range(t):
            if i == self.center_idx:
                # Center frame: extract full feature dimension
                feat = self.center_conv(x[:, i])
                features.append(feat)
            else:
                # History frames: extract reduced feature dimension
                feat = self.history_convs[history_idx](x[:, i])
                features.append(feat)
                history_idx += 1

        # Concatenate all features along channel dimension
        x_fused = torch.cat(features, dim=1)

        # Fuse features with multiple convolutions
        out_feature = self.fusion_conv(x_fused)

        # The rest of the forward pass is identical to the original SPAN model
        out_b1, _, _att1 = self.block_1(out_feature)
        out_b2, _, _att2 = self.block_2(out_b1)
        out_b3, _, _att3 = self.block_3(out_b2)

        out_b4, _, _att4 = self.block_4(out_b3)
        out_b5, _, _att5 = self.block_5(out_b4)
        out_b6, out_b5_2, _att6 = self.block_6(out_b5)

        out_b6 = self.conv_2(out_b6)
        out = self.conv_cat(torch.cat([out_feature, out_b6, out_b1, out_b5_2], 1))
        output = self.upsampler(out)

        return output