from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn

from ....architectures.__arch_helpers.padding import pad_to_multiple
from ....util import store_hyperparameters


def get_activation(activation: str = 'relu') -> nn.Module:
    """Get the specified activation layer.

    Args:
        activation (str): one of ``'relu'``, ``'leaky_relu'``, ``'elu'``, ``'gelu'``,
            ``'swish'``, 'efficient_swish'`` and ``'none'``. Default: ``'relu'``
    """
    assert activation in [
        'relu',
        'leaky_relu',
        'elu',
        'silu',
        'gelu',
        'none',
    ], f'Get unknown activation key {activation}'
    activation_dict = {
        'relu': nn.ReLU(inplace=True),
        'leaky_relu': nn.LeakyReLU(negative_slope=0.2, inplace=True),
        'elu': nn.ELU(alpha=1.0, inplace=True),
        'silu': nn.SiLU(inplace=True),
        'gelu': nn.GELU(),
        'none': nn.Identity(),
    }
    return activation_dict[activation]


def default_conv(
    in_channels: int, out_channels: int, kernel_size: int, bias: bool = True
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
    )


class MeanShift(nn.Conv2d):
    def __init__(
        self,
        rgb_range: int,
        rgb_mean: tuple[float, float, float],
        rgb_std: tuple[float, float, float],
        sign: float = -1,
    ) -> None:
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        assert self.bias is not None
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = False,
        bn: bool = True,
        act_mode: str | None = 'relu',
    ) -> None:
        m: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=(kernel_size // 2),
                stride=stride,
                bias=bias,
            )
        ]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act_mode is not None:
            m.append(get_activation(act_mode))
        super().__init__(*m)


class ResBlock(nn.Module):
    def __init__(
        self,
        conv: Callable[..., nn.Conv2d],
        n_feat: int,
        kernel_size: int,
        bias: bool = True,
        bn: bool = False,
        act_mode: str = 'relu',
        res_scale: float = 1,
    ) -> None:
        super().__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                m.append(get_activation(act_mode))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x: Tensor) -> Tensor:
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(
        self,
        conv: Callable[..., nn.Conv2d],
        scale: int,
        n_feat: int,
        bn: bool = False,
        act: nn.Module | None = None,
        bias: bool = True,
    ) -> None:
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feat))
                if act:
                    m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if act:
                m.append(act())
        else:
            raise NotImplementedError
        super().__init__(*m)


# Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 16) -> None:
        super().__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self,
        conv: Callable[..., nn.Conv2d],
        n_feat: int,
        kernel_size: int,
        reduction: int,
        bias: bool = True,
        bn: bool = False,
        act_mode: str = 'relu',
        res_scale: float = 1,
    ) -> None:
        super().__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(get_activation(act_mode))
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x: Tensor) -> Tensor:
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


# Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(
        self,
        conv: Callable[..., nn.Conv2d],
        n_feat: int,
        kernel_size: int,
        reduction: int,
        act_mode: str,
        res_scale: float,
        n_resblocks: int,
    ) -> None:
        super().__init__()
        modules_body = []
        modules_body: list[nn.Module] = [
            RCAB(
                conv,
                n_feat,
                kernel_size,
                reduction,
                bias=True,
                bn=False,
                act_mode=act_mode,
                res_scale=res_scale,
            )
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x: Tensor) -> Tensor:
        res = self.body(x)
        res += x
        return res


@store_hyperparameters()
# Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    hyperparameters = {}

    def __init__(
        self,
        *,
        scale: int = 4,
        n_resgroups: int = 10,
        n_resblocks: int = 20,
        n_feats: int = 64,
        n_colors: int = 3,
        rgb_range: int = 255,
        norm: bool = True,
        kernel_size: int = 3,
        reduction: int = 16,
        res_scale: float = 1,
        act_mode: str = 'relu',
        unshuffle_mod: bool = False,
        conv: Callable[..., nn.Conv2d] = default_conv,
    ) -> None:
        super().__init__()

        self.scale = scale

        if norm:
            # RGB mean for DIV2K
            self.rgb_range = rgb_range
            rgb_mean = (0.4488, 0.4371, 0.4040)
            rgb_std = (1.0, 1.0, 1.0)
            self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
            self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)
        else:
            self.rgb_range = 1
            self.sub_mean = nn.Identity()
            self.add_mean = nn.Identity()

        # define head module
        unshuffle_mod = unshuffle_mod and scale <= 2
        self.downscale_factor = 1
        if unshuffle_mod:
            self.downscale_factor = 4 // scale
            scale = 4
            modules_head = [
                nn.PixelUnshuffle(self.downscale_factor),
                conv(
                    n_colors * self.downscale_factor * self.downscale_factor,
                    n_feats,
                    kernel_size,
                ),
            ]
        else:
            modules_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        modules_body: list[nn.Module] = [
            ResidualGroup(
                conv,
                n_feats,
                kernel_size,
                reduction,
                act_mode=act_mode,
                res_scale=res_scale,
                n_resblocks=n_resblocks,
            )
            for _ in range(n_resgroups)
        ]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=None),
            conv(n_feats, n_colors, kernel_size),
        ]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def check_img_size(self, x: Tensor) -> Tensor:
        return pad_to_multiple(x, self.downscale_factor, mode='reflect')

    def forward(self, x: Tensor) -> Tensor:
        _b, _c, h, w = x.shape
        x = self.check_img_size(x)
        x *= self.rgb_range
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)
        return (x / self.rgb_range)[:, :, : h * self.scale, : w * self.scale]
