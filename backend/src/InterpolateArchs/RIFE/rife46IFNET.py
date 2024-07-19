import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .interpolate import interpolate
except ImportError:
    from torch.nn.functional import interpolate

from .warplayer import warp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def id():
    return "rife46"


def keys() -> list[str]:
    return [
        "module.block0.conv0.0.0.weight",
        "module.block0.conv0.0.0.bias",
        "module.block0.conv0.1.0.weight",
        "module.block0.conv0.1.0.bias",
        "module.block0.convblock.0.beta",
        "module.block0.convblock.0.conv.weight",
        "module.block0.convblock.0.conv.bias",
        "module.block0.convblock.1.beta",
        "module.block0.convblock.1.conv.weight",
        "module.block0.convblock.1.conv.bias",
        "module.block0.convblock.2.beta",
        "module.block0.convblock.2.conv.weight",
        "module.block0.convblock.2.conv.bias",
        "module.block0.convblock.3.beta",
        "module.block0.convblock.3.conv.weight",
        "module.block0.convblock.3.conv.bias",
        "module.block0.convblock.4.beta",
        "module.block0.convblock.4.conv.weight",
        "module.block0.convblock.4.conv.bias",
        "module.block0.convblock.5.beta",
        "module.block0.convblock.5.conv.weight",
        "module.block0.convblock.5.conv.bias",
        "module.block0.convblock.6.beta",
        "module.block0.convblock.6.conv.weight",
        "module.block0.convblock.6.conv.bias",
        "module.block0.convblock.7.beta",
        "module.block0.convblock.7.conv.weight",
        "module.block0.convblock.7.conv.bias",
        "module.block0.lastconv.0.weight",
        "module.block0.lastconv.0.bias",
        "module.block1.conv0.0.0.weight",
        "module.block1.conv0.0.0.bias",
        "module.block1.conv0.1.0.weight",
        "module.block1.conv0.1.0.bias",
        "module.block1.convblock.0.beta",
        "module.block1.convblock.0.conv.weight",
        "module.block1.convblock.0.conv.bias",
        "module.block1.convblock.1.beta",
        "module.block1.convblock.1.conv.weight",
        "module.block1.convblock.1.conv.bias",
        "module.block1.convblock.2.beta",
        "module.block1.convblock.2.conv.weight",
        "module.block1.convblock.2.conv.bias",
        "module.block1.convblock.3.beta",
        "module.block1.convblock.3.conv.weight",
        "module.block1.convblock.3.conv.bias",
        "module.block1.convblock.4.beta",
        "module.block1.convblock.4.conv.weight",
        "module.block1.convblock.4.conv.bias",
        "module.block1.convblock.5.beta",
        "module.block1.convblock.5.conv.weight",
        "module.block1.convblock.5.conv.bias",
        "module.block1.convblock.6.beta",
        "module.block1.convblock.6.conv.weight",
        "module.block1.convblock.6.conv.bias",
        "module.block1.convblock.7.beta",
        "module.block1.convblock.7.conv.weight",
        "module.block1.convblock.7.conv.bias",
        "module.block1.lastconv.0.weight",
        "module.block1.lastconv.0.bias",
        "module.block2.conv0.0.0.weight",
        "module.block2.conv0.0.0.bias",
        "module.block2.conv0.1.0.weight",
        "module.block2.conv0.1.0.bias",
        "module.block2.convblock.0.beta",
        "module.block2.convblock.0.conv.weight",
        "module.block2.convblock.0.conv.bias",
        "module.block2.convblock.1.beta",
        "module.block2.convblock.1.conv.weight",
        "module.block2.convblock.1.conv.bias",
        "module.block2.convblock.2.beta",
        "module.block2.convblock.2.conv.weight",
        "module.block2.convblock.2.conv.bias",
        "module.block2.convblock.3.beta",
        "module.block2.convblock.3.conv.weight",
        "module.block2.convblock.3.conv.bias",
        "module.block2.convblock.4.beta",
        "module.block2.convblock.4.conv.weight",
        "module.block2.convblock.4.conv.bias",
        "module.block2.convblock.5.beta",
        "module.block2.convblock.5.conv.weight",
        "module.block2.convblock.5.conv.bias",
        "module.block2.convblock.6.beta",
        "module.block2.convblock.6.conv.weight",
        "module.block2.convblock.6.conv.bias",
        "module.block2.convblock.7.beta",
        "module.block2.convblock.7.conv.weight",
        "module.block2.convblock.7.conv.bias",
        "module.block2.lastconv.0.weight",
        "module.block2.lastconv.0.bias",
        "module.block3.conv0.0.0.weight",
        "module.block3.conv0.0.0.bias",
        "module.block3.conv0.1.0.weight",
        "module.block3.conv0.1.0.bias",
        "module.block3.convblock.0.beta",
        "module.block3.convblock.0.conv.weight",
        "module.block3.convblock.0.conv.bias",
        "module.block3.convblock.1.beta",
        "module.block3.convblock.1.conv.weight",
        "module.block3.convblock.1.conv.bias",
        "module.block3.convblock.2.beta",
        "module.block3.convblock.2.conv.weight",
        "module.block3.convblock.2.conv.bias",
        "module.block3.convblock.3.beta",
        "module.block3.convblock.3.conv.weight",
        "module.block3.convblock.3.conv.bias",
        "module.block3.convblock.4.beta",
        "module.block3.convblock.4.conv.weight",
        "module.block3.convblock.4.conv.bias",
        "module.block3.convblock.5.beta",
        "module.block3.convblock.5.conv.weight",
        "module.block3.convblock.5.conv.bias",
        "module.block3.convblock.6.beta",
        "module.block3.convblock.6.conv.weight",
        "module.block3.convblock.6.conv.bias",
        "module.block3.convblock.7.beta",
        "module.block3.convblock.7.conv.weight",
        "module.block3.convblock.7.conv.bias",
        "module.block3.lastconv.0.weight",
        "module.block3.lastconv.0.bias",
    ]


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
            nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1), MyPixelShuffle(2)
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
    def __init__(self, scale=1, ensemble=False):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7, c=192)
        self.block1 = IFBlock(8 + 4, c=128)
        self.block2 = IFBlock(8 + 4, c=96)
        self.block3 = IFBlock(8 + 4, c=64)
        self.scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        self.ensemble = ensemble

    def forward(self, img0, img1, timestep, tenFlow_div, backwarp_tenGrid):
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        block = [self.block0, self.block1, self.block2, self.block3]
        for i in range(4):
            if flow is None:
                flow, mask = block[i](
                    torch.cat((img0[:, :3], img1[:, :3], timestep), 1),
                    None,
                    scale=self.scale_list[i],
                )
                if self.ensemble:
                    f1, m1 = block[i](
                        torch.cat((img1[:, :3], img0[:, :3], 1 - timestep), 1),
                        None,
                        scale=self.scale_list[i],
                    )
                    flow = (flow + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    mask = (mask + (-m1)) / 2
            else:
                f0, m0 = block[i](
                    torch.cat(
                        (warped_img0[:, :3], warped_img1[:, :3], timestep, mask), 1
                    ),
                    flow,
                    scale=self.scale_list[i],
                )
                if self.ensemble:
                    f1, m1 = block[i](
                        torch.cat(
                            (
                                warped_img1[:, :3],
                                warped_img0[:, :3],
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
            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2], tenFlow_div, backwarp_tenGrid)
            warped_img1 = warp(img1, flow[:, 2:4], tenFlow_div, backwarp_tenGrid)
            merged.append((warped_img0, warped_img1))
        mask_list[3] = torch.sigmoid(mask_list[3])
        return merged[3][0] * mask_list[3] + merged[3][1] * (1 - mask_list[3])
