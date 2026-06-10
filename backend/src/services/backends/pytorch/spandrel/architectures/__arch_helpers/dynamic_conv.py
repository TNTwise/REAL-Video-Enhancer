# type: ignore
import math
from typing import (
    TypeVar,
)

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, init
from torch.nn.init import trunc_normal_
from torch.nn.modules.utils import _pair

"""
28-Sep-21
https://github.com/TArdelean/DynamicConvolution/blob/master/dynamic_convolutions.py
https://github.com/TArdelean/DynamicConvolution/blob/master/models/common.py
MIT License

Copyright (c) 2024 Andrei-Timotei Ardelean,  Andreea Dogaru,  Alexey Larionov, Saian Protasov

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


T = TypeVar("T", bound=Module)


def normal_init(module, mean=0, std=1.0, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        trunc_normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class Conv2dWrapper(nn.Conv2d):
    """
    Wrapper for pytorch Conv2d class which can take additional parameters(like temperature) and ignores them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return super().forward(x)


class TempModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, temperature) -> torch.Tensor:
        return x


class BaseModel(TempModule):
    def __init__(self, ConvLayer):
        super().__init__()
        self.ConvLayer = ConvLayer


class TemperatureScheduler:
    def __init__(self, initial_value, final_value=None, final_epoch=None):
        self.initial_value = initial_value
        self.final_value = final_value or initial_value
        self.final_epoch = final_epoch or 1
        self.step = (
            0
            if self.final_epoch == 1
            else (final_value - initial_value) / (final_epoch - 1)
        )

    def get(self, crt_epoch=None):
        crt_epoch = crt_epoch or self.final_epoch
        return self.initial_value + (min(crt_epoch, self.final_epoch) - 1) * self.step


class Conv2dWrapper(nn.Conv2d):
    """
    Wrapper for pytorch Conv2d class which can take additional parameters(like temperature) and ignores them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return super().forward(x)


class TempModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, temperature) -> torch.Tensor:
        return x


class BaseModel(TempModule):
    def __init__(self, ConvLayer):
        super().__init__()
        self.ConvLayer = ConvLayer


class TemperatureScheduler:
    def __init__(self, initial_value, final_value=None, final_epoch=None):
        self.initial_value = initial_value
        self.final_value = final_value or initial_value
        self.final_epoch = final_epoch or 1
        self.step = (
            0
            if self.final_epoch == 1
            else (final_value - initial_value) / (final_epoch - 1)
        )

    def get(self, crt_epoch=None):
        crt_epoch = crt_epoch or self.final_epoch
        return self.initial_value + (min(crt_epoch, self.final_epoch) - 1) * self.step


class CustomSequential(TempModule):
    """Sequential container that supports passing temperature to TempModule"""

    def __init__(self, *args):
        super().__init__()
        self.layers = nn.ModuleList(args)

    def forward(self, x, temperature):
        for layer in self.layers:
            if isinstance(layer, TempModule):
                x = layer(x, temperature)
            else:
                x = layer(x)
        return x

    def __getitem__(self, idx):
        return CustomSequential(*list(self.layers[idx]))
        # if isinstance(idx, slice):
        #     return self.__class__(OrderedDict(list(self.layers.items())[idx]))
        # else:
        #     return self._get_item_by_idx(self.layers.values(), idx)


# Implementation inspired from
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/train.py#L38 and
# https://github.com/pytorch/pytorch/issues/7455
class SmoothNLLLoss(nn.Module):
    def __init__(self, smoothing=0.0, dim=-1):
        super().__init__()
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, prediction, target):
        with torch.no_grad():
            smooth_target = torch.zeros_like(prediction)
            n_class = prediction.size(self.dim)
            smooth_target.fill_(self.smoothing / (n_class - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        return torch.mean(torch.sum(-smooth_target * prediction, dim=self.dim))


class AttentionLayer(nn.Module):
    def __init__(self, c_dim, hidden_dim, nof_kernels):
        super().__init__()
        self.global_pooling = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.to_scores = nn.Sequential(
            nn.Linear(c_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, nof_kernels),
        )

    def forward(self, x, temperature=1):
        out = self.global_pooling(x)
        scores = self.to_scores(out)
        return F.softmax(scores / temperature, dim=-1)


class DynamicConvolution(TempModule):
    def __init__(
        self,
        nof_kernels,
        reduce,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        """
        Implementation of Dynamic convolution layer
        :param in_channels: number of input channels.
        :param out_channels: number of output channels.
        :param kernel_size: size of the kernel.
        :param groups: controls the connections between inputs and outputs.
        in_channels and out_channels must both be divisible by groups.
        :param nof_kernels: number of kernels to use.
        :param reduce: Refers to the size of the hidden layer in attention: hidden = in_channels // reduce
        :param bias: If True, convolutions also have a learnable bias
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.groups = groups
        self.conv_args = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
        }
        self.nof_kernels = nof_kernels
        self.attention = AttentionLayer(
            in_channels, max(1, in_channels // reduce), nof_kernels
        )
        self.kernel_size = _pair(kernel_size)
        self.kernels_weights = nn.Parameter(
            torch.Tensor(
                nof_kernels,
                out_channels,
                in_channels // self.groups,
                *self.kernel_size,
            ),
            requires_grad=True,
        )
        if bias:
            self.kernels_bias = nn.Parameter(
                torch.Tensor(nof_kernels, out_channels), requires_grad=True
            )
        else:
            self.register_parameter("kernels_bias", None)
        self.initialize_parameters()

    def initialize_parameters(self):
        for i_kernel in range(self.nof_kernels):
            init.kaiming_uniform_(self.kernels_weights[i_kernel], a=math.sqrt(5))
        if self.kernels_bias is not None:
            bound = 1 / math.sqrt(self.kernels_weights[0, 0].numel())
            nn.init.uniform_(self.kernels_bias, -bound, bound)

    def forward(self, x, temperature=1):
        batch_size = x.shape[0]

        alphas = self.attention(x, temperature)
        agg_weights = torch.sum(
            torch.mul(
                self.kernels_weights.unsqueeze(0),
                alphas.view(batch_size, -1, 1, 1, 1, 1),
            ),
            dim=1,
        )
        # Group the weights for each batch to conv2 all at once
        agg_weights = agg_weights.view(
            -1, *agg_weights.shape[-3:]
        )  # batch_size*out_c X in_c X kernel_size X kernel_size
        if self.kernels_bias is not None:
            agg_bias = torch.sum(
                torch.mul(
                    self.kernels_bias.unsqueeze(0),
                    alphas.view(batch_size, -1, 1),
                ),
                dim=1,
            )
            agg_bias = agg_bias.view(-1)
        else:
            agg_bias = None
        x_grouped = x.view(1, -1, *x.shape[-2:])  # 1 X batch_size*out_c X H X W

        out = F.conv2d(
            x_grouped,
            agg_weights,
            agg_bias,
            groups=self.groups * batch_size,
            **self.conv_args,
        )  # 1 X batch_size*out_C X H' x W'
        out = out.view(batch_size, -1, *out.shape[-2:])  # batch_size X out_C X H' x W'

        return out
