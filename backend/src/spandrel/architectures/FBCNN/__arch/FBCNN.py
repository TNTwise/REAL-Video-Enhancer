from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from spandrel.util import store_hyperparameters

"""
# --------------------------------------------
# Advanced nn.Sequential
# https://github.com/xinntao/BasicSR
# --------------------------------------------
"""


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
def conv(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="CBR",
    negative_slope=0.2,
):
    L = []
    for t in mode:
        if t == "C":
            L.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )
        elif t == "T":
            L.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )
        elif t == "B":
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == "I":
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == "R":
            L.append(nn.ReLU(inplace=True))
        elif t == "r":
            L.append(nn.ReLU(inplace=False))
        elif t == "L":
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == "l":
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == "2":
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == "3":
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == "4":
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == "U":
            L.append(nn.Upsample(scale_factor=2, mode="nearest"))
        elif t == "u":
            L.append(nn.Upsample(scale_factor=3, mode="nearest"))
        elif t == "v":
            L.append(nn.Upsample(scale_factor=4, mode="nearest"))
        elif t == "M":
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == "A":
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError("Undefined type: ".format())
    return sequential(*L)


# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# --------------------------------------------
class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        mode="CRC",
        negative_slope=0.2,
    ):
        super().__init__()

        assert in_channels == out_channels, "Only support in_channels==out_channels."
        if mode[0] in ["R", "L"]:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias,
            mode,
            negative_slope,
        )

    def forward(self, x):
        res = self.res(x)
        return x + res


# --------------------------------------------
# conv + subp (+ relu)
# --------------------------------------------
def upsample_pixelshuffle(
    in_channels=64,
    out_channels=3,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="2R",
    negative_slope=0.2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
        "4",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."
    up1 = conv(
        in_channels,
        out_channels * (int(mode[0]) ** 2),
        kernel_size,
        stride,
        padding,
        bias,
        mode="C" + mode,
        negative_slope=negative_slope,
    )
    return up1


# --------------------------------------------
# nearest_upsample + conv (+ R)
# --------------------------------------------
def upsample_upconv(
    in_channels=64,
    out_channels=3,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="2R",
    negative_slope=0.2,
):
    assert len(mode) < 4, "mode examples: 2, 2R, 2BR, 3, ..., 4BR"
    if mode[0] == "2":
        uc = "UC"
    elif mode[0] == "3":
        uc = "uC"
    elif mode[0] == "4":
        uc = "vC"
    else:
        raise ValueError(f"Wrong mode: {mode}. mode examples: 2, 2R, 2BR, 3, ..., 4BR")
    mode = mode.replace(mode[0], uc)
    up1 = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode=mode,
        negative_slope=negative_slope,
    )
    return up1


# --------------------------------------------
# convTranspose (+ relu)
# --------------------------------------------
def upsample_convtranspose(
    in_channels=64,
    out_channels=3,
    kernel_size=2,
    stride=2,
    padding=0,
    bias=True,
    mode="2R",
    negative_slope=0.2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
        "4",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], "T")
    up1 = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode,
        negative_slope,
    )
    return up1


"""
# --------------------------------------------
# Downsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# downsample_strideconv
# downsample_maxpool
# downsample_avgpool
# --------------------------------------------
"""


# --------------------------------------------
# strideconv (+ relu)
# --------------------------------------------
def downsample_strideconv(
    in_channels=64,
    out_channels=64,
    kernel_size=2,
    stride=2,
    padding=0,
    bias=True,
    mode="2R",
    negative_slope=0.2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
        "4",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], "C")
    down1 = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode,
        negative_slope,
    )
    return down1


# --------------------------------------------
# maxpooling + conv (+ relu)
# --------------------------------------------
def downsample_maxpool(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=0,
    bias=True,
    mode="2R",
    negative_slope=0.2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 3BR."
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], "MC")
    pool = conv(
        kernel_size=kernel_size_pool,
        stride=stride_pool,
        mode=mode[0],
        negative_slope=negative_slope,
    )
    pool_tail = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode=mode[1:],
        negative_slope=negative_slope,
    )
    return sequential(pool, pool_tail)


# --------------------------------------------
# averagepooling + conv (+ relu)
# --------------------------------------------
def downsample_avgpool(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="2R",
    negative_slope=0.2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 3BR."
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], "AC")
    pool = conv(
        kernel_size=kernel_size_pool,
        stride=stride_pool,
        mode=mode[0],
        negative_slope=negative_slope,
    )
    pool_tail = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode=mode[1:],
        negative_slope=negative_slope,
    )
    return sequential(pool, pool_tail)


class QFAttention(nn.Module):
    def __init__(
        self,
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        mode="CRC",
        negative_slope=0.2,
    ):
        super().__init__()

        assert in_channels == out_channels, "Only support in_channels==out_channels."
        if mode[0] in ["R", "L"]:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias,
            mode,
            negative_slope,
        )

    def forward(self, x, gamma, beta):
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        res = (gamma) * self.res(x) + beta
        return x + res


@store_hyperparameters()
class FBCNN(nn.Module):
    hyperparameters = {}

    def __init__(
        self,
        *,
        in_nc=3,
        out_nc=3,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode="R",
        downsample_mode="strideconv",
        upsample_mode="convtranspose",
    ):
        super().__init__()

        self.m_head = conv(in_nc, nc[0], bias=True, mode="C")
        self.nb = nb
        self.nc = nc
        # downsample
        if downsample_mode == "avgpool":
            downsample_block = downsample_avgpool
        elif downsample_mode == "maxpool":
            downsample_block = downsample_maxpool
        elif downsample_mode == "strideconv":
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError(
                f"downsample mode [{downsample_mode:s}] is not found"
            )

        self.m_down1 = sequential(
            *[
                ResBlock(nc[0], nc[0], bias=True, mode="C" + act_mode + "C")
                for _ in range(nb)
            ],
            downsample_block(nc[0], nc[1], bias=True, mode="2"),
        )
        self.m_down2 = sequential(
            *[
                ResBlock(nc[1], nc[1], bias=True, mode="C" + act_mode + "C")
                for _ in range(nb)
            ],
            downsample_block(nc[1], nc[2], bias=True, mode="2"),
        )
        self.m_down3 = sequential(
            *[
                ResBlock(nc[2], nc[2], bias=True, mode="C" + act_mode + "C")
                for _ in range(nb)
            ],
            downsample_block(nc[2], nc[3], bias=True, mode="2"),
        )

        self.m_body_encoder = sequential(
            *[
                ResBlock(nc[3], nc[3], bias=True, mode="C" + act_mode + "C")
                for _ in range(nb)
            ]
        )

        self.m_body_decoder = sequential(
            *[
                ResBlock(nc[3], nc[3], bias=True, mode="C" + act_mode + "C")
                for _ in range(nb)
            ]
        )

        # upsample
        if upsample_mode == "upconv":
            upsample_block = upsample_upconv
        elif upsample_mode == "pixelshuffle":
            upsample_block = upsample_pixelshuffle
        elif upsample_mode == "convtranspose":
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError(f"upsample mode [{upsample_mode:s}] is not found")

        self.m_up3 = nn.ModuleList(
            [
                upsample_block(nc[3], nc[2], bias=True, mode="2"),
                *[
                    QFAttention(nc[2], nc[2], bias=True, mode="C" + act_mode + "C")
                    for _ in range(nb)
                ],
            ]
        )

        self.m_up2 = nn.ModuleList(
            [
                upsample_block(nc[2], nc[1], bias=True, mode="2"),
                *[
                    QFAttention(nc[1], nc[1], bias=True, mode="C" + act_mode + "C")
                    for _ in range(nb)
                ],
            ]
        )

        self.m_up1 = nn.ModuleList(
            [
                upsample_block(nc[1], nc[0], bias=True, mode="2"),
                *[
                    QFAttention(nc[0], nc[0], bias=True, mode="C" + act_mode + "C")
                    for _ in range(nb)
                ],
            ]
        )

        self.m_tail = conv(nc[0], out_nc, bias=True, mode="C")

        self.qf_pred = sequential(
            *[
                ResBlock(nc[3], nc[3], bias=True, mode="C" + act_mode + "C")
                for _ in range(nb)
            ],
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(512, 512),
            nn.ReLU(),
            torch.nn.Linear(512, 512),
            nn.ReLU(),
            torch.nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        self.qf_embed = sequential(
            torch.nn.Linear(1, 512),
            nn.ReLU(),
            torch.nn.Linear(512, 512),
            nn.ReLU(),
            torch.nn.Linear(512, 512),
            nn.ReLU(),
        )

        self.to_gamma_3 = sequential(torch.nn.Linear(512, nc[2]), nn.Sigmoid())
        self.to_beta_3 = sequential(torch.nn.Linear(512, nc[2]), nn.Tanh())
        self.to_gamma_2 = sequential(torch.nn.Linear(512, nc[1]), nn.Sigmoid())
        self.to_beta_2 = sequential(torch.nn.Linear(512, nc[1]), nn.Tanh())
        self.to_gamma_1 = sequential(torch.nn.Linear(512, nc[0]), nn.Sigmoid())
        self.to_beta_1 = sequential(torch.nn.Linear(512, nc[0]), nn.Tanh())

    def forward(self, x, qf_input=None):
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body_encoder(x4)
        qf = self.qf_pred(x)
        x = self.m_body_decoder(x)
        qf_embedding = (
            self.qf_embed(qf_input) if qf_input is not None else self.qf_embed(qf)
        )
        gamma_3 = self.to_gamma_3(qf_embedding)
        beta_3 = self.to_beta_3(qf_embedding)

        gamma_2 = self.to_gamma_2(qf_embedding)
        beta_2 = self.to_beta_2(qf_embedding)

        gamma_1 = self.to_gamma_1(qf_embedding)
        beta_1 = self.to_beta_1(qf_embedding)

        x = x + x4
        x = self.m_up3[0](x)
        for i in range(self.nb):
            x = self.m_up3[i + 1](x, gamma_3, beta_3)

        x = x + x3

        x = self.m_up2[0](x)
        for i in range(self.nb):
            x = self.m_up2[i + 1](x, gamma_2, beta_2)
        x = x + x2

        x = self.m_up1[0](x)
        for i in range(self.nb):
            x = self.m_up1[i + 1](x, gamma_1, beta_1)

        x = x + x1
        x = self.m_tail(x)
        x = x[..., :h, :w]

        return x, qf
