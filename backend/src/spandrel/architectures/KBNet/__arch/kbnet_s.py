# type: ignore
import math

import torch
import torch.nn as nn
import torch.nn.init as init

from spandrel.util import store_hyperparameters

from ...__arch_helpers.padding import pad_to_multiple
from .kb_utils import KBAFunction, LayerNorm2d, SimpleGate


class KBBlock_s(nn.Module):
    def __init__(
        self, c, DW_Expand=2, FFN_Expand=2, nset=32, k=3, gc=4, lightweight=False
    ):
        super().__init__()
        self.k, self.c = k, c
        self.nset = nset
        dw_ch = int(c * DW_Expand)
        ffn_ch = int(FFN_Expand * c)

        self.g = c // gc
        self.w = nn.Parameter(torch.zeros(1, nset, c * c // self.g * self.k**2))
        self.b = nn.Parameter(torch.zeros(1, nset, c))
        self.init_p(self.w, self.b)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=c,
                out_channels=c,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )

        if not lightweight:
            self.conv11 = nn.Sequential(
                nn.Conv2d(
                    in_channels=c,
                    out_channels=c,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    groups=1,
                    bias=True,
                ),
                nn.Conv2d(
                    in_channels=c,
                    out_channels=c,
                    kernel_size=5,
                    padding=2,
                    stride=1,
                    groups=c // 4,
                    bias=True,
                ),
            )
        else:
            self.conv11 = nn.Sequential(
                nn.Conv2d(
                    in_channels=c,
                    out_channels=c,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    groups=1,
                    bias=True,
                ),
                nn.Conv2d(
                    in_channels=c,
                    out_channels=c,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    groups=c,
                    bias=True,
                ),
            )

        self.conv1 = nn.Conv2d(
            in_channels=c,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv21 = nn.Conv2d(
            in_channels=c,
            out_channels=c,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=c,
            bias=True,
        )

        interc = min(c, 32)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=c,
                out_channels=interc,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=interc,
                bias=True,
            ),
            SimpleGate(),
            nn.Conv2d(interc // 2, self.nset, 1, padding=0, stride=1),
        )

        self.conv211 = nn.Conv2d(in_channels=c, out_channels=self.nset, kernel_size=1)

        self.conv3 = nn.Conv2d(
            in_channels=dw_ch // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.conv4 = nn.Conv2d(
            in_channels=c,
            out_channels=ffn_ch,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_ch // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()

        self.ga1 = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
        self.attgamma = nn.Parameter(
            torch.zeros((1, self.nset, 1, 1)) + 1e-2, requires_grad=True
        )
        self.sg = SimpleGate()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)

    def init_p(self, weight, bias=None):
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(bias, -bound, bound)

    def KBA(self, x, att, selfk, selfg, selfb, selfw):
        return KBAFunction.apply(x, att, selfk, selfg, selfb, selfw)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        sca = self.sca(x)
        x1 = self.conv11(x)

        # KBA module
        att = self.conv2(x) * self.attgamma + self.conv211(x)
        uf = self.conv21(self.conv1(x))
        x = self.KBA(uf, att, self.k, self.g, self.b, self.w) * self.ga1 + uf
        x = x * x1 * sca

        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        # FFN
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        return y + x * self.gamma


@store_hyperparameters()
class KBNet_s(nn.Module):
    hyperparameters = {}

    def __init__(
        self,
        *,
        img_channel=3,
        width=64,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
        lightweight=False,
        ffn_scale=2,
    ):
        super().__init__()

        self.intro = nn.Conv2d(
            in_channels=img_channel,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        self.encoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.decoders = nn.ModuleList()

        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=img_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[
                        KBBlock_s(chan, FFN_Expand=ffn_scale, lightweight=lightweight)
                        for _ in range(num)
                    ]
                )
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[
                KBBlock_s(chan, FFN_Expand=ffn_scale, lightweight=lightweight)
                for _ in range(middle_blk_num)
            ]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[
                        KBBlock_s(chan, FFN_Expand=ffn_scale, lightweight=lightweight)
                        for _ in range(num)
                    ]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        _B, _C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        return pad_to_multiple(x, self.padder_size, mode="constant")
