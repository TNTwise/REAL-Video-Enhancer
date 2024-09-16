import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from spandrel.util import store_hyperparameters

from ...__arch_helpers.padding import pad_to_multiple


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)  # type: ignore

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)  # type: ignore

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class ISA(nn.Module):
    def __init__(self, dim, bias):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.data.shape
        x = x.view(b, c, -1).transpose(-1, -2)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.transpose(-1, -2)
        k = k.transpose(-1, -2)
        v = v.transpose(-1, -2)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.view(b, c, h, w)

        out = self.project_out(out)
        return out


class SDA(nn.Module):
    def __init__(self, n_feats, LayerNorm_type="WithBias"):
        super().__init__()
        i_feats = 2 * n_feats
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        self.DConvs = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 5, 1, 5 // 2, groups=n_feats),
            nn.Conv2d(
                n_feats,
                n_feats,
                7,
                stride=1,
                padding=(7 // 2) * 3,
                groups=n_feats,
                dilation=3,
            ),
            nn.Conv2d(n_feats, n_feats, 1, 1, 0),
        )

        self.proj_first = nn.Sequential(nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(nn.Conv2d(n_feats, n_feats, 1, 1, 0))
        self.dim = n_feats

    def forward(self, x):
        x = self.proj_first(x)
        a, x = torch.chunk(x, 2, dim=1)
        a = self.DConvs(a)
        x = self.proj_last(x * a) * self.scale

        return x


class ITL(nn.Module):
    def __init__(self, n_feats, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.attn = ISA(n_feats, bias)
        self.act = nn.Tanh()
        self.conv1 = nn.Conv2d(n_feats, n_feats, 1)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 1)

        self.ffn = FeedForward(n_feats, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.conv1(self.act(x)))
        x = x + self.ffn(self.conv2(self.act(x)))
        return x


class SAL(nn.Module):
    def __init__(self, n_feats, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.SDA = SDA(n_feats)
        self.ffn = FeedForward(n_feats, ffn_expansion_factor, bias)
        self.act = nn.Tanh()
        self.conv1 = nn.Conv2d(n_feats, n_feats, 1)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 1)

    def forward(self, x):
        x = x + self.SDA(self.conv1(self.act(x)))
        x = x + self.ffn(self.conv2(self.act(x)))
        return x


class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))

        super().__init__(*m)


class UFONE(nn.Module):
    def __init__(
        self,
        dim,
        ffn_expansion_factor,
        bias,
        LayerNorm_type,
        ITL_blocks,
        SAL_blocks,
        patch_size,
    ):
        super().__init__()
        ITL_body = [
            ITL(dim, ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(ITL_blocks)
        ]
        self.ITLs = nn.Sequential(*ITL_body)
        SAL_body = [
            SAL(dim, ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(SAL_blocks)
        ]
        self.SALs = nn.Sequential(*SAL_body)
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.data.shape
        local_features = x.view(
            B,
            C,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
        )
        local_features = (
            local_features.permute(0, 2, 4, 1, 3, 5)
            .contiguous()
            .view(-1, C, self.patch_size, self.patch_size)
        )
        local_features = self.ITLs(local_features)
        local_features = local_features.view(
            B,
            H // self.patch_size,
            W // self.patch_size,
            C,
            self.patch_size,
            self.patch_size,
        )
        global_features = (
            local_features.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
        )
        global_features = self.SALs(global_features)
        return global_features


@store_hyperparameters()
class DITN_Real(nn.Module):
    hyperparameters = {}

    def __init__(
        self,
        *,
        inp_channels=3,
        dim=60,
        ITL_blocks=4,
        SAL_blocks=4,
        UFONE_blocks=1,
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type="WithBias",
        patch_size=8,
        upscale=4,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.sft = nn.Conv2d(inp_channels, dim, 3, 1, 1)

        ## UFONE Block1
        UFONE_body = [
            UFONE(
                dim,
                ffn_expansion_factor,
                bias,
                LayerNorm_type,
                ITL_blocks,
                SAL_blocks,
                patch_size,
            )
            for _ in range(UFONE_blocks)
        ]
        self.UFONE = nn.Sequential(*UFONE_body)

        self.conv_after_body = nn.Conv2d(dim, dim, 3, 1, 1)
        self.upsample = UpsampleOneStep(upscale, dim, 3)
        self.dim = dim
        self.patch_sizes = [8, 8]
        self.scale = upscale
        self.SAL_blocks = SAL_blocks
        self.ITL_blocks = ITL_blocks

    def check_image_size(self, x):
        wsize = self.patch_sizes[0]
        for i in range(1, len(self.patch_sizes)):
            wsize = wsize * self.patch_sizes[i] // math.gcd(wsize, self.patch_sizes[i])
        return pad_to_multiple(x, wsize, mode="reflect")

    def forward(self, inp_img):
        _, _, old_h, old_w = inp_img.shape
        inp_img = self.check_image_size(inp_img)
        sft = self.sft(inp_img)

        local_features = self.UFONE(sft)

        local_features = self.conv_after_body(local_features)
        out_dec_level1 = self.upsample(local_features + sft)

        return out_dec_level1[:, :, 0 : old_h * self.scale, 0 : old_w * self.scale]
