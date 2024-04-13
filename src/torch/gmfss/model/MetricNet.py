import torch
import torch.nn as nn
import torch.nn.functional as F

from .gmflow.geometry import forward_backward_consistency_check
from .util import MyPReLU

torch.fx.wrap('backwarp')
torch.fx.wrap('forward_backward_consistency_check')

backwarp_tenGrid = {}


def backwarp(tenIn, tenflow):
    if str(tenflow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(start=-1.0, end=1.0, steps=tenflow.shape[3], dtype=tenflow.dtype, device=tenflow.device).view(1, 1, 1, -1).repeat(1, 1, tenflow.shape[2], 1)
        tenVer = torch.linspace(start=-1.0, end=1.0, steps=tenflow.shape[2], dtype=tenflow.dtype, device=tenflow.device).view(1, 1, -1, 1).repeat(1, 1, 1, tenflow.shape[3])

        backwarp_tenGrid[str(tenflow.shape)] = torch.cat([tenHor, tenVer], 1)
    # end

    tenflow = torch.cat([tenflow[:, 0:1, :, :] / ((tenIn.shape[3] - 1.0) / 2.0), tenflow[:, 1:2, :, :] / ((tenIn.shape[2] - 1.0) / 2.0)], 1)

    return torch.nn.functional.grid_sample(input=tenIn, grid=(backwarp_tenGrid[str(tenflow.shape)] + tenflow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)


class MetricNet(nn.Module):
    def __init__(self):
        super(MetricNet, self).__init__()
        self.metric_in = nn.Conv2d(14, 64, 3, 1, 1)
        self.metric_net1 = nn.Sequential(
            MyPReLU(),
            nn.Conv2d(64, 64, 3, 1, 1)
        )
        self.metric_net2 = nn.Sequential(
            MyPReLU(),
            nn.Conv2d(64, 64, 3, 1, 1)
        )
        self.metric_net3 = nn.Sequential(
            MyPReLU(),
            nn.Conv2d(64, 64, 3, 1, 1)
        )
        self.metric_out = nn.Sequential(
            MyPReLU(),
            nn.Conv2d(64, 2, 3, 1, 1)
        )

    def forward(self, img0, img1, flow01, flow10):
        metric0 = F.l1_loss(img0, backwarp(img1, flow01), reduction='none').mean([1], True)
        metric1 = F.l1_loss(img1, backwarp(img0, flow10), reduction='none').mean([1], True)

        fwd_occ, bwd_occ = forward_backward_consistency_check(flow01, flow10)

        flow01 = torch.cat([flow01[:, 0:1, :, :] / ((flow01.shape[3] - 1.0) / 2.0), flow01[:, 1:2, :, :] / ((flow01.shape[2] - 1.0) / 2.0)], 1)
        flow10 = torch.cat([flow10[:, 0:1, :, :] / ((flow10.shape[3] - 1.0) / 2.0), flow10[:, 1:2, :, :] / ((flow10.shape[2] - 1.0) / 2.0)], 1)

        img = torch.cat((img0, img1), 1)
        metric = torch.cat((-metric0, -metric1), 1)
        flow = torch.cat((flow01, flow10), 1)
        occ = torch.cat((fwd_occ.unsqueeze(1), bwd_occ.unsqueeze(1)), 1)

        feat = self.metric_in(torch.cat((img, metric, flow, occ), 1))
        feat = self.metric_net1(feat) + feat
        feat = self.metric_net2(feat) + feat
        feat = self.metric_net3(feat) + feat
        metric = self.metric_out(feat)

        metric = torch.tanh(metric) * 10

        return metric[:, :1], metric[:, 1:2]
