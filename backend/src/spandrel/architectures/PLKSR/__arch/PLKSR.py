from functools import partial
from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from spandrel.util import store_hyperparameters


# Since Pytorch's interleave is not supported by CoreML, we use this function instead for mobile conversion
def repeat_interleave(x, n):
    x = x.unsqueeze(2)
    x = x.repeat(1, 1, n, 1, 1)
    x = x.reshape(x.shape[0], x.shape[1] * n, x.shape[3], x.shape[4])
    return x


class CCM(nn.Sequential):
    "Convolutional Channel Mixer"

    def __init__(self, dim: int):
        super().__init__(
            nn.Conv2d(dim, dim * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1, 1, 0),
        )
        trunc_normal_(self[-1].weight, std=0.02)


class ICCM(nn.Sequential):
    "Inverted Convolutional Channel Mixer"

    def __init__(self, dim: int):
        super().__init__(
            nn.Conv2d(dim, dim * 2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 3, 1, 1),
        )
        trunc_normal_(self[-1].weight, std=0.02)


class DCCM(nn.Sequential):
    "Doubled Convolutional Channel Mixer"

    def __init__(self, dim: int):
        super().__init__(
            nn.Conv2d(dim, dim * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 3, 1, 1),
        )
        trunc_normal_(self[-1].weight, std=0.02)


class PLKConv2d(nn.Module):
    "Partial Large Kernel Convolutional Layer"

    def __init__(self, dim, kernel_size, with_idt):
        super().__init__()
        self.with_idt = with_idt
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2)
        trunc_normal_(self.conv.weight, std=0.02)
        self.idx = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x1, x2 = torch.split(x, [self.idx, x.size(1) - self.idx], dim=1)
            if self.with_idt:
                x1 = self.conv(x1) + x1
            else:
                x1 = self.conv(x1)
            return torch.cat([x1, x2], dim=1)
        else:
            if self.with_idt:
                x[:, : self.idx] = x[:, : self.idx] + self.conv(x[:, : self.idx])
            else:
                x[:, : self.idx] = self.conv(x[:, : self.idx])
            return x


class RectSparsePLKConv2d(nn.Module):
    "Rectangular Sparse Partial Large Kernel Convolutional Layer (SLaK style)"

    def __init__(self, dim, kernel_size):
        super().__init__()
        self.idx = dim
        m = kernel_size
        n = kernel_size // 3
        self.mn_conv = nn.Conv2d(dim, dim, (m, n), 1, (m // 2, n // 2))
        self.nm_conv = nn.Conv2d(dim, dim, (n, m), 1, (n // 2, m // 2))
        self.nn_conv = nn.Conv2d(dim, dim, (n, n), 1, (n // 2, n // 2))

        trunc_normal_(self.mn_conv.weight, std=0.02)
        trunc_normal_(self.nm_conv.weight, std=0.02)
        trunc_normal_(self.nn_conv.weight, std=0.02)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # No reparametrization since this is for a ablative study
        if self.training:
            x1, x2 = x[:, : self.idx], x[:, self.idx :]
            x1 = self.mn_conv(x1) + self.nm_conv(x1) + self.nn_conv(x1)
            return torch.cat([x1, x2], dim=1)

        else:
            x[:, : self.idx] = (
                self.mn_conv(x[:, : self.idx])
                + self.nm_conv(x[:, : self.idx])
                + self.nn_conv(x[:, : self.idx])
            )
            return x


class SparsePLKConv2d(nn.Module):
    "Sparse Partial Large Kernel Convolutional Layer (RepLKNet and UniRepLKNet style)"

    def __init__(
        self,
        dim,
        max_kernel_size,
        sub_kernel_sizes,
        dilations,
        use_max_kernel,
        with_idt,
    ):
        super().__init__()
        self.use_max_kernel = use_max_kernel
        self.max_kernel_size = max_kernel_size
        for k, d in zip(sub_kernel_sizes, dilations):
            m_k = self._calc_rep_kernel_size(k, d)
            if m_k > self.max_kernel_size:
                self.max_kernel_size = m_k
        self.with_idt = with_idt

        convs = [
            nn.Conv2d(
                dim, dim, sub_kernel_size, 1, (sub_kernel_size // 2) * d, dilation=d
            )
            for sub_kernel_size, d in zip(sub_kernel_sizes, dilations)
        ]
        if use_max_kernel:
            convs.append(
                nn.Conv2d(dim, dim, self.max_kernel_size, 1, self.max_kernel_size // 2)
            )
        self.convs = nn.ModuleList(convs)
        for m in self.convs:
            trunc_normal_(m.weight, std=0.02)
        self.idx = dim
        self.is_convert = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_convert:
            x[:, : self.idx, :, :] = self.conv(x[:, : self.idx, :, :])
            return x
        else:
            x1, x2 = torch.split(x, [self.idx, x.size(1) - self.idx], dim=1)
            if self.with_idt:
                out = x1
            else:
                out = 0.0
            for conv in self.convs:
                out = out + conv(x1)
            return torch.cat([out, x2], dim=1)  # type: ignore

    @staticmethod
    def _calc_rep_kernel_size(ks, dilation):
        return (ks - 1) * dilation + 1

    @staticmethod
    def _get_origin_kernel(kernel, dilation=1, p=0):
        I = torch.ones((1, 1, 1, 1)).to(kernel.device)  # noqa: E741
        if kernel.size(1) == 1:  # Depth-wise Convolution
            dilated = F.conv_transpose2d(kernel, I, stride=dilation)
        else:
            slices = []  # Dense or Group
            for i in range(kernel.size(1)):
                dilated = F.conv_transpose2d(
                    kernel[:, i : i + 1, :, :], I, stride=dilation
                )
                slices.append(dilated)
            dilated = torch.cat(slices, dim=1)

        # Pad boundary
        if p != 0:
            dilated = F.pad(dilated, (p, p, p, p))
        return dilated

    @staticmethod
    def _dwc_to_dense(kernel):
        n_groups = kernel.size(0)

        kernels = []
        for g in range(n_groups):
            kernels.append(
                torch.cat(
                    [
                        kernel[g : (g + 1)]
                        if g == i
                        else torch.zeros_like(kernel[g : (g + 1)])
                        for i in range(n_groups)
                    ],
                    dim=1,
                )
            )
        return torch.cat(kernels, dim=0)


class EA(nn.Module):
    "Element-wise Attention"

    def __init__(self, dim: int):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.Sigmoid())
        trunc_normal_(self.f[0].weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.f(x)


class PLKBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        # CCM Rep options
        ccm_type: Literal["CCM", "ICCM", "DCCM"],
        # LK Options
        max_kernel_size: int,
        split_ratio: float,
        lk_type: Literal["PLK", "SparsePLK", "RectSparsePLK"] = "PLK",
        # Sparse Rep options
        use_max_kernel: bool = False,
        sparse_kernels: Sequence[int] = [5, 5, 5],
        sparse_dilations: Sequence[int] = [2, 3, 4],
        with_idt: bool = False,
        # EA ablation
        use_ea: bool = True,
    ):
        super().__init__()

        # Local Texture
        if ccm_type == "CCM":
            self.channe_mixer = CCM(dim)
        elif ccm_type == "ICCM":
            self.channe_mixer = ICCM(dim)
        elif ccm_type == "DCCM":
            self.channe_mixer = DCCM(dim)
        else:
            raise ValueError(f"Unknown CCM type: {ccm_type}")

        # Long-range Dependency
        pdim = int(dim * split_ratio)
        if lk_type == "PLK":
            self.lk = PLKConv2d(pdim, max_kernel_size, with_idt)
        elif lk_type == "SparsePLK":
            self.lk = SparsePLKConv2d(
                pdim,
                max_kernel_size,
                sparse_kernels,
                sparse_dilations,
                use_max_kernel,
                with_idt,
            )
        elif lk_type == "RectSparsePLK":
            self.lk = RectSparsePLKConv2d(pdim, max_kernel_size)
        else:
            raise ValueError(f"Unknown LK type: {lk_type}")

        # Instance-dependent modulation
        if use_ea:
            self.attn = EA(dim)
        else:
            self.attn = nn.Identity()

        # Refinement
        self.refine = nn.Conv2d(dim, dim, 1, 1, 0)
        trunc_normal_(self.refine.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = x
        x = self.channe_mixer(x)
        x = self.lk(x)
        x = self.attn(x)
        x = self.refine(x)
        return x + x_skip


@store_hyperparameters()
class PLKSR(nn.Module):
    hyperparameters = {}

    def __init__(
        self,
        *,
        dim: int = 64,
        n_blocks: int = 28,
        upscaling_factor: int = 4,
        # CCM options
        ccm_type: Literal["CCM", "ICCM", "DCCM"] = "CCM",
        # LK Options
        kernel_size: int = 17,
        split_ratio: float = 0.25,
        lk_type: Literal["PLK", "SparsePLK", "RectSparsePLK"] = "PLK",
        # LK Rep options
        use_max_kernel: bool = False,
        sparse_kernels: Sequence[int] = [5, 5, 5, 5],
        sparse_dilations: Sequence[int] = [1, 2, 3, 4],
        with_idt: bool = False,
        # EA ablation
        use_ea: bool = True,
    ):
        super().__init__()

        self.feats = nn.Sequential(
            *[nn.Conv2d(3, dim, 3, 1, 1)]
            + [
                PLKBlock(
                    dim,
                    ccm_type,
                    kernel_size,
                    split_ratio,
                    lk_type,
                    use_max_kernel,
                    sparse_kernels,
                    sparse_dilations,
                    with_idt,
                    use_ea,
                )
                for _ in range(n_blocks)
            ]
            + [nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1)]
        )
        trunc_normal_(self.feats[0].weight, std=0.02)
        trunc_normal_(self.feats[-1].weight, std=0.02)

        self.to_img = nn.PixelShuffle(upscaling_factor)

        self.repeat_op = partial(
            torch.repeat_interleave, repeats=upscaling_factor**2, dim=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feats(x) + self.repeat_op(x)
        x = self.to_img(x)
        return x
