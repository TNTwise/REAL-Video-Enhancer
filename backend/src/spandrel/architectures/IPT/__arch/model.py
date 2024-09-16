# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>


import torch

from .ipt import IPT


class Model:
    def __init__(
        self,
        model: IPT,
        crop_batch_size: int = 64,
    ):
        super().__init__()

        self.model: IPT = model
        self.crop_batch_size: int = crop_batch_size

    def forward(self, x: torch.Tensor, scale_idx: int):
        self.model.set_scale(scale_idx)
        return self.forward_chop(x)

    def forward_chop(self, x: torch.Tensor, shave=12):
        batchsize = self.crop_batch_size
        h, w = x.size()[-2:]
        padsize = int(self.model.patch_size)
        shave = int(self.model.patch_size / 2)

        scale = self.model.scale[self.model.scale_idx]

        h_cut = (h - padsize) % (int(shave / 2))
        w_cut = (w - padsize) % (int(shave / 2))

        x_unfold = (
            torch.nn.functional.unfold(x, padsize, stride=int(shave / 2))
            .transpose(0, 2)
            .contiguous()
        )

        x_hw_cut = x[..., (h - padsize) :, (w - padsize) :]
        y_hw_cut = self.model.forward(x_hw_cut)

        x_h_cut = x[..., (h - padsize) :, :]
        x_w_cut = x[..., :, (w - padsize) :]
        y_h_cut = self.cut_h(
            x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize
        )
        y_w_cut = self.cut_w(
            x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize
        )

        x_h_top = x[..., :padsize, :]
        x_w_top = x[..., :, :padsize]
        y_h_top = self.cut_h(
            x_h_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize
        )
        y_w_top = self.cut_w(
            x_w_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize
        )

        x_unfold = x_unfold.view(x_unfold.size(0), -1, padsize, padsize)
        y_unfold = []

        x_range = x_unfold.size(0) // batchsize + (x_unfold.size(0) % batchsize != 0)
        for i in range(x_range):
            y_unfold.append(
                self.model(x_unfold[i * batchsize : (i + 1) * batchsize, ...])
            )
        y_unfold = torch.cat(y_unfold, dim=0)

        y = torch.nn.functional.fold(
            y_unfold.view(y_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            ((h - h_cut) * scale, (w - w_cut) * scale),
            padsize * scale,
            stride=int(shave / 2 * scale),
        )

        y[..., : padsize * scale, :] = y_h_top
        y[..., :, : padsize * scale] = y_w_top

        y_unfold = y_unfold[
            ...,
            int(shave / 2 * scale) : padsize * scale - int(shave / 2 * scale),
            int(shave / 2 * scale) : padsize * scale - int(shave / 2 * scale),
        ].contiguous()
        y_inter = torch.nn.functional.fold(
            y_unfold.view(y_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            ((h - h_cut - shave) * scale, (w - w_cut - shave) * scale),
            padsize * scale - shave * scale,
            stride=int(shave / 2 * scale),
        )

        y_ones = torch.ones(y_inter.shape, dtype=y_inter.dtype).to(x)
        divisor = torch.nn.functional.fold(
            torch.nn.functional.unfold(
                y_ones, padsize * scale - shave * scale, stride=int(shave / 2 * scale)
            ),
            ((h - h_cut - shave) * scale, (w - w_cut - shave) * scale),
            padsize * scale - shave * scale,
            stride=int(shave / 2 * scale),
        )

        y_inter = y_inter / divisor

        y[
            ...,
            int(shave / 2 * scale) : (h - h_cut) * scale - int(shave / 2 * scale),
            int(shave / 2 * scale) : (w - w_cut) * scale - int(shave / 2 * scale),
        ] = y_inter

        y = torch.cat(
            [
                y[..., : y.size(2) - int((padsize - h_cut) / 2 * scale), :],
                y_h_cut[..., int((padsize - h_cut) / 2 * scale + 0.5) :, :],
            ],
            dim=2,
        )
        y_w_cat = torch.cat(
            [
                y_w_cut[..., : y_w_cut.size(2) - int((padsize - h_cut) / 2 * scale), :],
                y_hw_cut[..., int((padsize - h_cut) / 2 * scale + 0.5) :, :],
            ],
            dim=2,
        )
        y = torch.cat(
            [
                y[..., :, : y.size(3) - int((padsize - w_cut) / 2 * scale)],
                y_w_cat[..., :, int((padsize - w_cut) / 2 * scale + 0.5) :],
            ],
            dim=3,
        )
        return y

    def cut_h(
        self,
        x_h_cut: torch.Tensor,
        h: int,
        w: int,
        h_cut: int,
        w_cut: int,
        padsize: int,
        shave: int,
        scale: int,
        batchsize: int,
    ):
        x_h_cut_unfold = (
            torch.nn.functional.unfold(x_h_cut, padsize, stride=int(shave / 2))
            .transpose(0, 2)
            .contiguous()
        )

        x_h_cut_unfold = x_h_cut_unfold.view(
            x_h_cut_unfold.size(0), -1, padsize, padsize
        )
        x_range = x_h_cut_unfold.size(0) // batchsize + (
            x_h_cut_unfold.size(0) % batchsize != 0
        )
        y_h_cut_unfold = []
        for i in range(x_range):
            y_h_cut_unfold.append(
                self.model(x_h_cut_unfold[i * batchsize : (i + 1) * batchsize, ...])
            )
        y_h_cut_unfold = torch.cat(y_h_cut_unfold, dim=0)

        y_h_cut = torch.nn.functional.fold(
            y_h_cut_unfold.view(y_h_cut_unfold.size(0), -1, 1)
            .transpose(0, 2)
            .contiguous(),
            (padsize * scale, (w - w_cut) * scale),
            padsize * scale,
            stride=int(shave / 2 * scale),
        )
        y_h_cut_unfold = y_h_cut_unfold[
            ..., :, int(shave / 2 * scale) : padsize * scale - int(shave / 2 * scale)
        ].contiguous()
        y_h_cut_inter = torch.nn.functional.fold(
            y_h_cut_unfold.view(y_h_cut_unfold.size(0), -1, 1)
            .transpose(0, 2)
            .contiguous(),
            (padsize * scale, (w - w_cut - shave) * scale),
            (padsize * scale, padsize * scale - shave * scale),
            stride=int(shave / 2 * scale),
        )

        y_ones = torch.ones(y_h_cut_inter.shape, dtype=y_h_cut_inter.dtype).to(x_h_cut)
        divisor = torch.nn.functional.fold(
            torch.nn.functional.unfold(
                y_ones,
                (padsize * scale, padsize * scale - shave * scale),
                stride=int(shave / 2 * scale),
            ),
            (padsize * scale, (w - w_cut - shave) * scale),
            (padsize * scale, padsize * scale - shave * scale),
            stride=int(shave / 2 * scale),
        )
        y_h_cut_inter = y_h_cut_inter / divisor

        y_h_cut[
            ...,
            :,
            int(shave / 2 * scale) : (w - w_cut) * scale - int(shave / 2 * scale),
        ] = y_h_cut_inter
        return y_h_cut

    def cut_w(
        self,
        x_w_cut: torch.Tensor,
        h: int,
        w: int,
        h_cut: int,
        w_cut: int,
        padsize: int,
        shave: int,
        scale: int,
        batchsize: int,
    ):
        x_w_cut_unfold = (
            torch.nn.functional.unfold(x_w_cut, padsize, stride=int(shave / 2))
            .transpose(0, 2)
            .contiguous()
        )

        x_w_cut_unfold = x_w_cut_unfold.view(
            x_w_cut_unfold.size(0), -1, padsize, padsize
        )
        x_range = x_w_cut_unfold.size(0) // batchsize + (
            x_w_cut_unfold.size(0) % batchsize != 0
        )
        y_w_cut_unfold = []
        for i in range(x_range):
            y_w_cut_unfold.append(
                self.model(x_w_cut_unfold[i * batchsize : (i + 1) * batchsize, ...])
            )
        y_w_cut_unfold = torch.cat(y_w_cut_unfold, dim=0)

        y_w_cut = torch.nn.functional.fold(
            y_w_cut_unfold.view(y_w_cut_unfold.size(0), -1, 1)
            .transpose(0, 2)
            .contiguous(),
            ((h - h_cut) * scale, padsize * scale),
            padsize * scale,
            stride=int(shave / 2 * scale),
        )
        y_w_cut_unfold = y_w_cut_unfold[
            ..., int(shave / 2 * scale) : padsize * scale - int(shave / 2 * scale), :
        ].contiguous()
        y_w_cut_inter = torch.nn.functional.fold(
            y_w_cut_unfold.view(y_w_cut_unfold.size(0), -1, 1)
            .transpose(0, 2)
            .contiguous(),
            ((h - h_cut - shave) * scale, padsize * scale),
            (padsize * scale - shave * scale, padsize * scale),
            stride=int(shave / 2 * scale),
        )

        y_ones = torch.ones(y_w_cut_inter.shape, dtype=y_w_cut_inter.dtype).to(x_w_cut)
        divisor = torch.nn.functional.fold(
            torch.nn.functional.unfold(
                y_ones,
                (padsize * scale - shave * scale, padsize * scale),
                stride=int(shave / 2 * scale),
            ),
            ((h - h_cut - shave) * scale, padsize * scale),
            (padsize * scale - shave * scale, padsize * scale),
            stride=int(shave / 2 * scale),
        )
        y_w_cut_inter = y_w_cut_inter / divisor

        y_w_cut[
            ...,
            int(shave / 2 * scale) : (h - h_cut) * scale - int(shave / 2 * scale),
            :,
        ] = y_w_cut_inter
        return y_w_cut
