from typing import Literal

import torch.nn as nn

from spandrel.util import store_hyperparameters

from ...__arch_helpers.dpir_basic_block import conv, sequential

"""
# --------------------------------------------
# DnCNN (20 conv layers)
# FDnCNN (20 conv layers)
# --------------------------------------------
# References:
@article{zhang2017beyond,
    title={Beyond a gaussian denoiser: Residual learning of deep cnn for image   denoising},
    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang,   Lei},
    journal={IEEE Transactions on Image Processing},
    volume={26},
    number={7},
    pages={3142--3155},
    year={2017},
    publisher={IEEE}
}
@article{zhang2018ffdnet,
    title={FFDNet: Toward a fast and flexible solution for CNN-based image denoising},
    author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
    journal={IEEE Transactions on Image Processing},
    volume={27},
    number={9},
    pages={4608--4622},
    year={2018},
    publisher={IEEE}
}
# --------------------------------------------
"""


# --------------------------------------------
# DnCNN
# --------------------------------------------
@store_hyperparameters()
class DnCNN(nn.Module):
    hyperparameters = {}

    def __init__(
        self,
        *,
        in_nc=1,
        out_nc=1,
        nc=64,
        nb=17,
        act_mode="BR",
        mode: Literal["DnCNN", "FDnCNN"] = "DnCNN",
    ):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super().__init__()
        assert (
            "R" in act_mode or "L" in act_mode
        ), "Examples of activation function: R, L, BR, BL, IR, IL"
        bias = True

        self.mode = mode
        if mode == "DnCNN":
            assert in_nc == out_nc, "DnCNN only supports in_nc == out_nc"

        m_head = conv(in_nc, nc, mode="C" + act_mode[-1], bias=bias)
        m_body = [conv(nc, nc, mode="C" + act_mode, bias=bias) for _ in range(nb - 2)]
        m_tail = conv(nc, out_nc, mode="C", bias=bias)

        self.model = sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        if self.mode == "DnCNN":
            n = self.model(x)
            return x - n
        else:
            return self.model(x)
