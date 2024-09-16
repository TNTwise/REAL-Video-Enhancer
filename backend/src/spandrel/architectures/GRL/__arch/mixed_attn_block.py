from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import GRLConfig
from .ops import bchw_to_bhwc, bchw_to_blc, blc_to_bchw, blc_to_bhwc


class CPB_MLP(nn.Sequential):
    def __init__(self, in_channels, out_channels, channels=512):
        m = [
            nn.Linear(in_channels, channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels, out_channels, bias=False),
        ]
        super().__init__(*m)


class SeparableConv(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, bias, args: GRLConfig
    ):
        m: list[torch.nn.Module] = [
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                groups=in_channels,
                bias=bias,
            )
        ]
        if args.separable_conv_act:
            m.append(nn.GELU())
        m.append(nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias))
        super().__init__(*m)


class QKVProjection(nn.Module):
    def __init__(self, dim, qkv_bias, proj_type, args: GRLConfig):
        super().__init__()
        self.proj_type = proj_type
        if proj_type == "linear":
            self.body = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.body = SeparableConv(dim, dim * 3, 3, 1, qkv_bias, args)

    def forward(self, x, x_size):
        if self.proj_type == "separable_conv":
            x = blc_to_bchw(x, x_size)
        x = self.body(x)
        if self.proj_type == "separable_conv":
            x = bchw_to_blc(x)
        return x


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.reduction = nn.Linear(4 * in_dim, out_dim, bias=False)

    def forward(self, x, x_size):
        """
        x: B, H*W, C
        """
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)

        return x


class AnchorLinear(nn.Module):
    r"""Linear anchor projection layer
    Args:
        dim (int): Number of input channels.
    """

    def __init__(self, in_channels, out_channels, down_factor, pooling_mode, bias):
        super().__init__()
        self.down_factor = down_factor
        if pooling_mode == "maxpool":
            self.pooling = nn.MaxPool2d(down_factor, down_factor)
        elif pooling_mode == "avgpool":
            self.pooling = nn.AvgPool2d(down_factor, down_factor)
        self.reduction = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x, x_size):
        """
        x: B, H*W, C
        """
        x = blc_to_bchw(x, x_size)
        x = bchw_to_blc(self.pooling(x))
        x = blc_to_bhwc(self.reduction(x), [s // self.down_factor for s in x_size])  # type: ignore
        return x


class AnchorProjection(nn.Module):
    def __init__(
        self,
        dim: int,
        proj_type: str,
        one_stage: bool,
        anchor_window_down_factor: int,
        args: GRLConfig,
    ):
        super().__init__()
        self.proj_type = proj_type
        self.body = nn.ModuleList([])
        if one_stage:
            if proj_type == "patchmerging":
                m = PatchMerging(dim, dim // 2)
            elif proj_type == "conv2d":
                kernel_size = anchor_window_down_factor + 1
                stride = anchor_window_down_factor
                padding = kernel_size // 2
                m = nn.Conv2d(dim, dim // 2, kernel_size, stride, padding)
            elif proj_type == "separable_conv":
                kernel_size = anchor_window_down_factor + 1
                stride = anchor_window_down_factor
                m = SeparableConv(dim, dim // 2, kernel_size, stride, True, args)
            elif proj_type.find("pool") >= 0:
                m = AnchorLinear(
                    dim, dim // 2, anchor_window_down_factor, proj_type, True
                )
            else:
                raise ValueError(f"Unsupported anchor projection type {proj_type}")
            self.body.append(m)
        else:
            for i in range(int(math.log2(anchor_window_down_factor))):
                cin = dim if i == 0 else dim // 2
                if proj_type == "patchmerging":
                    m = PatchMerging(cin, dim // 2)
                elif proj_type == "conv2d":
                    m = nn.Conv2d(cin, dim // 2, 3, 2, 1)
                elif proj_type == "separable_conv":
                    m = SeparableConv(cin, dim // 2, 3, 2, True, args)
                else:
                    raise ValueError(f"Unsupported anchor projection type {proj_type}")
                self.body.append(m)

    def forward(self, x, x_size):
        if self.proj_type.find("conv") >= 0:
            x = blc_to_bchw(x, x_size)
            for m in self.body:
                x = m(x)
            x = bchw_to_bhwc(x)
        elif self.proj_type.find("pool") >= 0:
            for m in self.body:
                x = m(x, x_size)
        else:
            for i, m in enumerate(self.body):
                x = m(x, [s // 2**i for s in x_size])
            x = blc_to_bhwc(x, [s // 2 ** (i + 1) for s in x_size])  # type: ignore
        return x


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        reduction (int): Channel reduction factor. Default: 16.
    """

    def __init__(self, num_feat, reduction=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // reduction, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // reduction, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=4, reduction=18):
        super().__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, reduction),
        )

    def forward(self, x, x_size):
        x = self.cab(blc_to_bchw(x, x_size).contiguous())
        return bchw_to_blc(x)
