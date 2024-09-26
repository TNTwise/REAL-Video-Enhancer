import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from .warplayer import warp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),        
        nn.LeakyReLU(0.2, True)
    )

def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, True)
    )
    
class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x, feat=False):
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)
        if feat:
            return [x0, x1, x2, x3]
        return x3

class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1\
)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
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
            nn.ConvTranspose2d(c, 4*13, 4, 2, 1),
            nn.PixelShuffle(2)
        )

    def forward(self, x, flow=None, scale=1):
        x = interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        feat = tmp[:, 5:]
        return flow, mask, feat


class IFNet(nn.Module):
    def __init__(
        self,
        scale=1.0,
        ensemble=False,
        dtype=torch.float32,
        device="cuda",
        width=1920,
        height=1080,
        backwarp_tenGrid=None,
        tenFlow_div=None,
    ):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7+8, c=192)
        self.block1 = IFBlock(8+4+8+8, c=128)
        self.block2 = IFBlock(8+4+8+8, c=96)
        self.block3 = IFBlock(8+4+8+8, c=64)
        self.block4 = IFBlock(8+4+8+8, c=32)
        self.encode = Head()
        self.device = device
        self.dtype = dtype
        self.scaleList = [16/scale,8 / scale, 4 / scale, 2 / scale, 1 / scale]
        self.ensemble = ensemble
        self.width = width
        self.height = height
        self.backWarp = backwarp_tenGrid
        self.tenFlow = tenFlow_div

        self.paddedHeight = backwarp_tenGrid.shape[2]
        self.paddedWidth = backwarp_tenGrid.shape[3]

        self.blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]

    def forward(self, img0, img1, timeStep, f0, f1):
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        for i in range(5):
            if flow is None:
                flow, mask, feat = self.blocks[i](
                    torch.cat((img0[:, :3], img1[:, :3], f0, f1, timeStep), 1),
                    None,
                    scale=self.scaleList[i],
                )
            else:
                wf0 = warp(f0, flow[:, :2], self.tenFlow, self.backWarp)
                wf1 = warp(f1, flow[:, 2:4], self.tenFlow, self.backWarp)
                fd, m0, feat = self.blocks[i](
                    torch.cat(
                        (
                            warped_img0[:, :3],
                            warped_img1[:, :3],
                            wf0,
                            wf1,
                            timeStep,
                            mask,
                            feat,
                        ),
                        1,
                    ),
                    flow,
                    scale=self.scaleList[i],
                )
                mask = m0
                flow = flow + fd
            warped_img0 = warp(img0, flow[:, :2], self.tenFlow, self.backWarp)
            warped_img1 = warp(img1, flow[:, 2:4], self.tenFlow, self.backWarp)
        mask = torch.sigmoid(mask)
        return (
            (warped_img0 * mask + warped_img1 * (1 - mask))[
                :, :, : self.height, : self.width
            ][0]
            .permute(1, 2, 0)
            .mul(255)
            .float()
        )
class IFNetV2(nn.Module):
    def __init__(
        self,
        scale=1.0,
        ensemble=False,
        dtype=torch.float32,
        device="cuda",
        width=1920,
        height=1080,
        backwarp_tenGrid=None,
        tenFlow_div=None,
    ):
        super(IFNetV2, self).__init__()
        self.block0 = IFBlock(7+8, c=192)
        self.block1 = IFBlock(8+4+8+8, c=128)
        self.block2 = IFBlock(8+4+8+8, c=96)
        self.block3 = IFBlock(8+4+8+8, c=64)
        self.block4 = IFBlock(8+4+8+8, c=32)
        self.encode = Head()
        self.device = device
        self.dtype = dtype
        self.scaleList = [16/scale,8 / scale, 4 / scale, 2 / scale, 1 / scale]
        self.ensemble = ensemble
        self.width = width
        self.height = height
        self.backWarp = backwarp_tenGrid
        self.tenFlow = tenFlow_div

        self.paddedHeight = backwarp_tenGrid.shape[2]
        self.paddedWidth = backwarp_tenGrid.shape[3]

        self.blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]

    def forward(self, img0, img1, timestep, f0, f1):
        # cant be cached
        h, w = img0.shape[2], img0.shape[3]
        imgs = torch.cat([img0, img1], dim=1)
        imgs_2 = torch.reshape(imgs, (2, 3, self.paddedHeight, self.paddedWidth))
        fs = torch.cat([f0, f1], dim=1)
        fs_2 = torch.reshape(fs, (2, 4, self.paddedHeight, self.paddedWidth))
       
        imgs_fs_2 = torch.cat((imgs_2, fs_2), 1)
        warped_img0 = img0
        warped_img1 = img1
        flows = None
        flows = None
        for block, scale in zip(self.blocks, self.scaleList):
            if flows is None:
                
                temp = torch.cat((imgs, fs, timestep), 1)
                flows, mask, feat = block(temp, scale=scale)
            else:
                
                temp = torch.cat((wimg, wf, timestep, mask, feat, (flows * (1 / scale) if scale != 1 else flows)), 1)
                fds, mask, feat = block(temp, scale=scale)

                flows = flows + fds

            precomp = (self.backWarp + flows.reshape((2, 2, h, w)) * self.tenFlow).permute(0, 2, 3, 1)
            if scale == 1:
                warped_imgs = torch.nn.functional.grid_sample(imgs_2, precomp, mode='bilinear', padding_mode='border', align_corners=True)
            else:
                warps = torch.nn.functional.grid_sample(imgs_fs_2, precomp, mode='bilinear', padding_mode='border', align_corners=True)
                wimg, wf = torch.split(warps, [3, 4], dim=1)
                wimg = torch.reshape(wimg, (1, 6, h, w))
                wf = torch.reshape(wf, (1, 8, h, w))
           
        mask = torch.sigmoid(mask)
        warped_img0, warped_img1 = torch.split(warped_imgs, [1, 1])
        return (
            (warped_img0 * mask + warped_img1 * (1 - mask))[
                :, :, : self.height, : self.width
            ][0]
            .permute(1, 2, 0)
            .mul(255)
            .float()
        )