import torch.nn as nn

from spandrel.util import store_hyperparameters

from ...__arch_helpers.dpir_basic_block import (
    ResBlock,
    conv,
    downsample_avgpool,
    downsample_maxpool,
    downsample_strideconv,
    sequential,
    upsample_convtranspose,
    upsample_pixelshuffle,
    upsample_upconv,
)


@store_hyperparameters()
class DRUNet(nn.Module):
    hyperparameters = {}

    def __init__(
        self,
        *,
        in_nc=1,
        out_nc=1,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode="R",
        downsample_mode="strideconv",
        upsample_mode="convtranspose",
    ):
        super().__init__()

        self.m_head = conv(in_nc, nc[0], bias=False, mode="C")

        # downsample
        if downsample_mode == "avgpool":
            downsample_block = downsample_avgpool
        elif downsample_mode == "maxpool":
            downsample_block = downsample_maxpool
        elif downsample_mode == "strideconv":
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError(
                f"downsample mode [{downsample_mode:s}] is not found"
            )

        self.m_down1 = sequential(
            *[
                ResBlock(nc[0], nc[0], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ],
            downsample_block(nc[0], nc[1], bias=False, mode="2"),
        )
        self.m_down2 = sequential(
            *[
                ResBlock(nc[1], nc[1], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ],
            downsample_block(nc[1], nc[2], bias=False, mode="2"),
        )
        self.m_down3 = sequential(
            *[
                ResBlock(nc[2], nc[2], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ],
            downsample_block(nc[2], nc[3], bias=False, mode="2"),
        )

        self.m_body = sequential(
            *[
                ResBlock(nc[3], nc[3], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ]
        )

        # upsample
        if upsample_mode == "upconv":
            upsample_block = upsample_upconv
        elif upsample_mode == "pixelshuffle":
            upsample_block = upsample_pixelshuffle
        elif upsample_mode == "convtranspose":
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError(f"upsample mode [{upsample_mode:s}] is not found")

        self.m_up3 = sequential(
            upsample_block(nc[3], nc[2], bias=False, mode="2"),
            *[
                ResBlock(nc[2], nc[2], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ],
        )
        self.m_up2 = sequential(
            upsample_block(nc[2], nc[1], bias=False, mode="2"),
            *[
                ResBlock(nc[1], nc[1], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ],
        )
        self.m_up1 = sequential(
            upsample_block(nc[1], nc[0], bias=False, mode="2"),
            *[
                ResBlock(nc[0], nc[0], bias=False, mode="C" + act_mode + "C")
                for _ in range(nb)
            ],
        )

        self.m_tail = conv(nc[0], out_nc, bias=False, mode="C")

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        return x
