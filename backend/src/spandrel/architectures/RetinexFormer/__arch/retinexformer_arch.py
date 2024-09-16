from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from spandrel.util import store_hyperparameters  # type: ignore
from spandrel.util.timm import trunc_normal_


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: Callable):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class Illumination_Estimator(nn.Module):
    def __init__(
        self,
        n_fea_middle: int,
        n_fea_in: int = 4,
        n_fea_out: int = 3,
    ):  # __init__部分是内部属性，而forward的输入才是外部输入
        super().__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle,
            n_fea_middle,
            kernel_size=5,
            padding=2,
            bias=True,
            groups=n_fea_in,
        )

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w

        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        mean_c = img.mean(dim=1).unsqueeze(1)
        input = torch.cat([img, mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


class IG_MSA(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans  # illu_fea: b,c,h,w -> b,h,w,c
        q, k, v, illu_attn = (
            rearrange(t, "b n (h d) -> b h n d", h=self.num_heads)
            for t in (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2))
        )
        v = v * illu_attn
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = k @ q.transpose(-2, -1)  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(
            0, 2, 3, 1
        )
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


class IGAB(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleList(
                    [
                        IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                        PreNorm(dim, FeedForward(dim=dim)),
                    ]
                )
            )

    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for attn, ff in self.blocks:  # type: ignore
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super().__init__()
        self.dim = dim
        self.level = level

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(
                nn.ModuleList(
                    [
                        IGAB(
                            dim=dim_level,
                            num_blocks=num_blocks[i],
                            dim_head=dim,
                            heads=dim_level // dim,
                        ),
                        nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                        nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                    ]
                )
            )
            dim_level *= 2

        # Bottleneck
        self.bottleneck = IGAB(
            dim=dim_level,
            dim_head=dim,
            heads=dim_level // dim,
            num_blocks=num_blocks[-1],
        )

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(
                            dim_level,
                            dim_level // 2,
                            stride=2,
                            kernel_size=2,
                            padding=0,
                            output_padding=0,
                        ),
                        nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                        IGAB(
                            dim=dim_level // 2,
                            num_blocks=num_blocks[level - 1 - i],
                            dim_head=dim,
                            heads=(dim_level // 2) // dim,
                        ),
                    ]
                )
            )
            dim_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:  # type: ignore
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        """
        x:          [b,c,h,w]         x是feature, 不是image
        illu_fea:   [b,c,h,w]
        return out: [b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        for IGAB, FeaDownSample, IlluFeaDownsample in self.encoder_layers:  # type: ignore
            fea = IGAB(fea, illu_fea)  # bchw
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # Bottleneck
        fea = self.bottleneck(fea, illu_fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):  # type: ignore
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level - 1 - i]
            fea = LeWinBlcok(fea, illu_fea)

        # Mapping
        out = self.mapping(fea) + x

        return out


class RetinexFormer_Single_Stage(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        n_feat=31,
        level=2,
        num_blocks=[1, 1, 1],
    ):
        super().__init__()
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(
            in_dim=in_channels,
            out_dim=out_channels,
            dim=n_feat,
            level=level,
            num_blocks=num_blocks,
        )  #### 将 Denoiser 改为 img2img

    def forward(self, img):
        # img:        b,c=3,h,w

        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        illu_fea, illu_map = self.estimator(img)
        input_img = img * illu_map + img
        output_img = self.denoiser(input_img, illu_fea)

        return output_img


@store_hyperparameters()
class RetinexFormer(nn.Module):
    hyperparameters = {}

    def __init__(
        self,
        *,
        in_channels=3,
        out_channels=3,
        n_feat=40,
        stage=3,
        num_blocks=[1, 1, 1],
    ):
        super().__init__()
        self.stage = stage

        modules_body = [
            RetinexFormer_Single_Stage(
                in_channels=in_channels,
                out_channels=out_channels,
                n_feat=n_feat,
                level=2,
                num_blocks=num_blocks,
            )
            for _ in range(stage)
        ]

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        out = self.body(x)

        return out
