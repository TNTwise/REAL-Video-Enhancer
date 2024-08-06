import torch
import torch.nn as nn

try:
    from .interpolate import interpolate
except ImportError:
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


def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        ),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, True),
    )


class MyPixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(MyPixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        b, c, hh, hw = x.size()
        out_channel = c // (self.upscale_factor**2)
        h = hh * self.upscale_factor
        w = hw * self.upscale_factor
        x_view = x.view(
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
        x = interpolate(
            x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
        )
        if flow is not None:
            flow = (
                interpolate(
                    flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
                )
                / scale
            )
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        return flow, mask


class IFNet(nn.Module):
    def __init__(
        self,
        scale=1.0,
        ensemble=False,
        dtype=torch.float32,
        device="cuda",
    ):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7 + 8, c=192)
        self.block1 = IFBlock(8 + 4 + 8, c=128)
        self.block2 = IFBlock(8 + 4 + 8, c=96)
        self.block3 = IFBlock(8 + 4 + 8, c=64)
        self.encode = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ConvTranspose2d(16, 4, 4, 2, 1)
        )
        self.device = device
        self.dtype = dtype
        self.scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        self.ensemble = ensemble

        # self.contextnet = Contextnet()
        # self.unet = Unet()

    def forward(self, img0, img1, timestep, tenFlow_div, backwarp_tenGrid):
        # cant be cached
        h, w = img0.shape[2], img0.shape[3]
        tenFlow_div = tenFlow_div.reshape(1, 2, 1, 1)
        imgs = torch.cat([img0, img1], dim=1)
        imgs_2 = torch.reshape(imgs, (2, 3, h, w))
        fs_2 = self.encode(imgs_2)
        fs = torch.reshape(fs_2, (1, 16, h, w))
        if self.ensemble:
            fs_rev = torch.cat(torch.split(fs, [8, 8], dim=1)[::-1], dim=1)
            imgs_rev = torch.cat([img1, img0], dim=1)
        warped_img0 = img0
        warped_img1 = img1
        flows = None
        mask = None
        blocks = [self.block0, self.block1, self.block2, self.block3]
        for block, scale in zip(blocks, self.scale_list):
            if flows is None:
                if self.ensemble:
                    temp = torch.cat((imgs, fs, timestep), 1)
                    temp_ = torch.cat((imgs_rev, fs_rev, 1 - timestep), 1)
                    flowss, masks = block(torch.cat((temp, temp_), 0), scale=scale)
                    flows, flows_ = torch.split(flowss, [1, 1], dim=0)
                    mask, mask_ = torch.split(masks, [1, 1], dim=0)
                    flows = (
                        flows
                        + torch.cat(torch.split(flows_, [2, 2], dim=1)[::-1], dim=1)
                    ) / 2
                    mask = (mask - mask_) / 2

                    flows_rev = torch.cat(
                        torch.split(flows, [2, 2], dim=1)[::-1], dim=1
                    )
                else:
                    temp = torch.cat((imgs, fs, timestep), 1)
                    flows, mask = block(temp, scale=scale)
            else:
                if self.ensemble:
                    temp = torch.cat(
                        (
                            wimg,
                            wf,
                            timestep,
                            mask,
                            (flows * (1 / scale) if scale != 1 else flows),
                        ),
                        1,
                    )
                    temp_ = torch.cat(
                        (
                            wimg_rev,
                            wf_rev,
                            1 - timestep,
                            -mask,
                            (flows_rev * (1 / scale) if scale != 1 else flows_rev),
                        ),
                        1,
                    )
                    fdss, masks = block(torch.cat((temp, temp_), 0), scale=scale)
                    fds, fds_ = torch.split(fdss, [1, 1], dim=0)
                    mask, mask_ = torch.split(masks, [1, 1], dim=0)
                    fds = (
                        fds + torch.cat(torch.split(fds_, [2, 2], dim=1)[::-1], dim=1)
                    ) / 2
                    mask = (mask - mask_) / 2
                else:
                    temp = torch.cat(
                        (
                            wimg,
                            wf,
                            timestep,
                            mask,
                            (flows * (1 / scale) if scale != 1 else flows),
                        ),
                        1,
                    )
                    fds, mask = block(temp, scale=scale)

                flows = flows + fds

                if self.ensemble:
                    flows_rev = torch.cat(
                        torch.split(flows, [2, 2], dim=1)[::-1], dim=1
                    )
            precomp = (
                (backwarp_tenGrid + flows.reshape((2, 2, h, w)) * tenFlow_div)
                .permute(0, 2, 3, 1)
                .to(dtype=self.dtype)
            )
            if scale == 1:
                warped_imgs = torch.nn.functional.grid_sample(
                    imgs_2,
                    precomp,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
            else:
                warps = torch.nn.functional.grid_sample(
                    torch.cat(
                        (imgs_2.to(dtype=self.dtype), fs_2.to(dtype=self.dtype)), 1
                    ).to(dtype=self.dtype),
                    precomp,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                wimg, wf = torch.split(warps, [3, 8], dim=1)
                wimg = torch.reshape(wimg, (1, 6, h, w))
                wf = torch.reshape(wf, (1, 16, h, w))
                if self.ensemble:
                    wimg_rev = torch.cat(torch.split(wimg, [3, 3], dim=1)[::-1], dim=1)
                    wf_rev = torch.cat(torch.split(wf, [8, 8], dim=1)[::-1], dim=1)
        mask = torch.sigmoid(mask)
        warped_img0, warped_img1 = torch.split(warped_imgs, [1, 1])
        return warped_img0 * mask + warped_img1 * (1 - mask)
