import torch
import torch.nn as nn
import torch.nn.functional as F

from spandrel.util import store_hyperparameters


# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# CCM
class CCM(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
        )

    def forward(self, x):
        return self.ccm(x)


# SAFM
class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Spatial Weighting
        self.mfr = nn.ModuleList(
            [
                nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim)
                for _ in range(self.n_levels)
            ]
        )

        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2**i, w // 2**i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode="nearest")
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out


class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        # Multiscale Block
        self.safm = SAFM(dim)
        # Feedforward layer
        self.ccm = CCM(dim, ffn_scale)

    def forward(self, x):
        x = self.safm(self.norm1(x)) + x
        x = self.ccm(self.norm2(x)) + x
        return x


class BasicLayer(nn.Module):
    def __init__(self, num_layer, dim, ffn_scale=2.0):
        super().__init__()

        self.layers = nn.Sequential(
            *[AttBlock(dim, ffn_scale) for _ in range(num_layer)]
        )
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        return self.conv(self.layers(x)) + x


@store_hyperparameters()
class SAFMN_BCIE(nn.Module):
    hyperparameters = {}

    def __init__(
        self, *, dim: int, n_blocks=6, num_layers=6, ffn_scale=2.0, upscaling_factor=2
    ):
        super().__init__()

        self.to_feat = nn.Sequential(
            nn.PixelUnshuffle(upscaling_factor),
            nn.Conv2d(3 * upscaling_factor**2, dim, 3, 1, 1),
        )

        self.feats = nn.Sequential(
            *[BasicLayer(num_layers, dim, ffn_scale) for _ in range(n_blocks)]
        )

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, dim * upscaling_factor**2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor),
            nn.Conv2d(dim, 3, 3, 1, 1),
        )

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        x = self.to_img(x)
        return x
