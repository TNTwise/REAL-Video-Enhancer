import torch
from torch import nn
from torch.nn.init import trunc_normal_

from ....architectures.__arch_helpers.dysample import DySample
from ....util import store_hyperparameters
from ....util.timm.__drop import DropPath


class GPS(nn.Module):
    """Geo ensemble PixelShuffle"""

    def __init__(
        self,
        dim,
        scale,
        out_ch=3,
        # Own parameters
        kernel_size: int = 3,
    ):
        super().__init__()
        self.in_to_k = nn.Conv2d(
            dim, scale * scale * out_ch * 8, kernel_size, 1, kernel_size // 2
        )
        self.ps = nn.PixelShuffle(scale)

    def forward(self, x):
        rgb = self._geo_ensemble(x)
        rgb = self.ps(rgb)
        return rgb

    def _geo_ensemble(self, x):
        x = self.in_to_k(x)
        x = x.reshape(x.shape[0], 8, -1, x.shape[-2], x.shape[-1])
        x = x.mean(dim=1)
        return x


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvBlock(nn.Module):
    r"""https://github.com/joshyZhou/AST/blob/main/model.py#L22"""

    def __init__(self, in_channel: int, out_channel: int, strides: int = 1):
        super().__init__()
        self.strides = strides
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=3,
                stride=strides,
                padding=1,
            ),
            nn.Mish(),
            nn.Conv2d(
                out_channel,
                out_channel,
                kernel_size=3,
                stride=strides,
                padding=1,
            ),
            nn.Mish(),
        )
        self.conv11 = nn.Conv2d(
            in_channel, out_channel, kernel_size=1, stride=strides, padding=0
        )

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out


class GatedCNNBlock(nn.Module):
    r"""
    modernized mambaout main unit
    https://github.com/yuweihao/MambaOut/blob/main/models/mambaout.py#L119
    """

    def __init__(
        self,
        dim: int,
        expansion_ratio: float = 8 / 3,
        conv_ratio: float = 1.0,
        kernel_size: int = 7,
        drop_path: float = 0.5,
    ):
        super().__init__()
        self.norm = LayerNorm(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 3, 1, 1)

        self.act = nn.Mish()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = [hidden, hidden - conv_channels, conv_channels]

        self.conv = nn.Conv2d(
            conv_channels,
            conv_channels,
            kernel_size,
            1,
            kernel_size // 2,
            groups=conv_channels,
        )
        self.fc2 = nn.Conv2d(hidden, dim, 3, 1, 1)
        self.drop_path = (
            DropPath(drop_path)
            if drop_path > 0.0 or not self.training
            else nn.Identity()
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
        c = self.conv(c)
        x = self.act(self.fc2(self.act(g) * torch.cat((i, c), dim=1)))
        x = self.drop_path(x)
        return x + (shortcut - 0.5)


@store_hyperparameters()
class MoSR(nn.Module):
    """Mamba Out Super-Resolution"""

    hyperparameters = {}

    def __init__(
        self,
        *,
        in_ch: int = 3,
        out_ch: int = 3,
        upscale: int = 4,
        n_block: int = 24,
        dim: int = 64,
        upsampler: str = "ps",  # "ps" "dys" "gps"
        drop_path: float = 0.0,
        kernel_size: int = 7,
        expansion_ratio: float = 1.5,
        conv_ratio: float = 1.0,
    ):
        super().__init__()
        if upsampler in ["ps", "gps"]:
            out_ch = in_ch
        dp_rates = [x.item() for x in torch.linspace(0, drop_path, n_block)]
        self.gblocks = nn.Sequential(
            *[nn.Conv2d(in_ch, dim, 3, 1, 1)]
            + [
                GatedCNNBlock(
                    dim=dim,
                    expansion_ratio=expansion_ratio,
                    kernel_size=kernel_size,
                    conv_ratio=conv_ratio,
                    drop_path=dp_rates[index],
                )
                for index in range(n_block)
            ]
            + [
                nn.Conv2d(dim, dim * 2, 3, 1, 1),
                nn.Mish(),
                nn.Conv2d(dim * 2, dim, 3, 1, 1),
                nn.Mish(),
                nn.Conv2d(dim, dim, 1, 1),
            ]
        )

        self.shortcut = ConvBlock(in_ch, dim)

        if upsampler == "ps":
            self.upsampler = nn.Sequential(
                nn.Conv2d(dim, out_ch * (upscale**2), 3, 1, 1),
                nn.PixelShuffle(upscale),
            )
        elif upsampler == "gps":
            self.upsampler = GPS(dim, upscale, out_ch)
        elif upsampler == "dys":
            self.upsampler = DySample(dim, out_ch, upscale)
        else:
            raise ValueError(
                f'upsampler: {upsampler} not supported, choose one of these options: \
                ["ps", "gps", "dys"]'
            )

    def forward(self, x):
        x = self.gblocks(x) + (self.shortcut(x) - 0.5)
        return self.upsampler(x)
