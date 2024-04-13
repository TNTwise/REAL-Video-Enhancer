import torch.nn as nn

from .util import MyPReLU


class FeatureNet(nn.Module):
    """The quadratic model"""
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.block1 = nn.Sequential(
            MyPReLU(),
            nn.Conv2d(3, 64, 3, 2, 1),
            MyPReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.block2 = nn.Sequential(
            MyPReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            MyPReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
        )
        self.block3 = nn.Sequential(
            MyPReLU(),
            nn.Conv2d(128, 192, 3, 2, 1),
            MyPReLU(),
            nn.Conv2d(192, 192, 3, 1, 1),
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        return x1, x2, x3
