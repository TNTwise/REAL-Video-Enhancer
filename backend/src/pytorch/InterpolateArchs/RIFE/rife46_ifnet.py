"""
MIT License

Copyright (c) 2024 Hzwer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.nn.functional import interpolate



def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.LeakyReLU(0.2, True),
    )


class MyPixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(MyPixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        b, c, hh, hw = input.size()
        out_channel = c // (self.upscale_factor**2)
        h = hh * self.upscale_factor
        w = hw * self.upscale_factor
        x_view = input.view(
            b, out_channel, self.upscale_factor, self.upscale_factor, hh, hw
        )
        return x_view.permute(0, 1, 4, 2, 5, 3).reshape(b, out_channel, h, w)


class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1), nn.PixelShuffle(2)
        )

    def forward(self, x, flow=None, scale=1):
        x = interpolate(x, scale_factor=1.0 / scale, mode="bilinear")
        if flow is not None:
            flow = (
                interpolate(flow, scale_factor=1.0 / scale, mode="bilinear")
                * 1.0
                / scale
            )
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = interpolate(tmp, scale_factor=scale, mode="bilinear")
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        return flow, mask


class IFNet(nn.Module):
    def __init__(
        self,
        scale=1.0,
        ensemble=False,
        dtype=torch.float32,
        device: torch.device = torch.device("cuda"),
        width=1920,
        height=1080,
    ):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7, c=192)
        self.block1 = IFBlock(8 + 4, c=128)
        self.block2 = IFBlock(8 + 4, c=96)
        self.block3 = IFBlock(8 + 4, c=64)
        scale = scale
        self.scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        self.ensemble = ensemble
        self.dtype = dtype
        self.device = device
        self.width = width
        self.height = height
        self.block = [self.block0, self.block1, self.block2, self.block3]
        from ..util.warplayer import warp

        self.warp = warp

    def forward(self, img0, img1, timestep, tenFlow_div, backwarp_tenGrid, scale=None):
        img0 = img0.clamp(0.,1.)
        img1 = img1.clamp(0.,1.)
        if scale is not None:
            self.scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None

        for i in range(4):
            if flow is None:
                flow, mask = self.block[i](
                    torch.cat((img0, img1, timestep), 1),
                    None,
                    scale=self.scale_list[i],
                )
                if self.ensemble:
                    f1, m1 = self.block[i](
                        torch.cat((img1, img0, 1 - timestep), 1),
                        None,
                        scale=self.scale_list[i],
                    )
                    flow = (flow + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    mask = (mask + (-m1)) / 2
            else:
                f0, m0 = self.block[i](
                    torch.cat((warped_img0, warped_img1, timestep, mask), 1),
                    flow,
                    scale=self.scale_list[i],
                )
                if self.ensemble:
                    f1, m1 = self.block[i](
                        torch.cat(
                            (
                                warped_img1,
                                warped_img0,
                                1 - timestep,
                                -mask,
                            ),
                            1,
                        ),
                        torch.cat((flow[:, 2:4], flow[:, :2]), 1),
                        scale=self.scale_list[i],
                    )
                    f0 = (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    m0 = (m0 + (-m1)) / 2
                flow = flow + f0
                mask = mask + m0
            latest_mask = mask
            warped_img0 = self.warp(img0, flow[:, :2], tenFlow_div, backwarp_tenGrid)
            warped_img1 = self.warp(img1, flow[:, 2:4], tenFlow_div, backwarp_tenGrid)

        temp = torch.sigmoid(latest_mask)
        frame = warped_img0 * temp + warped_img1 * (1 - temp)
        return frame
