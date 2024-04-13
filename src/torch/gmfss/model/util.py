import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class MyPixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(MyPixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        b, c, hh, hw = input.size()
        out_channel = c // (self.upscale_factor**2)
        h = hh * self.upscale_factor
        w = hw * self.upscale_factor
        x_view = input.view(b, out_channel, self.upscale_factor, self.upscale_factor, hh, hw)
        return x_view.permute(0, 1, 4, 2, 5, 3).reshape(b, out_channel, h, w)


class MyPReLU(nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        super(MyPReLU, self).__init__()
        self.weight = Parameter(torch.empty(num_parameters).fill_(init))

    def forward(self, input):
        return F.relu(input) - self.weight.reshape(1, -1, 1, 1) * F.relu(-input)
