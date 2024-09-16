from math import pi

import torch
import torch.nn as nn


class RGB_HVI(nn.Module):
    def __init__(self):
        super().__init__()
        self.density_k = torch.nn.Parameter(
            torch.full([1], 0.2)
        )  # k is reciprocal to the paper mentioned
        self.gated = False
        self.gated2 = False
        self.alpha = 1.0
        self.this_k = 0

    def HVIT(self, img):
        eps = 1e-8
        device = img.device
        dtypes = img.dtype
        hue = (
            torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(device).to(dtypes)
        )
        value = img.max(1)[0].to(dtypes)
        img_min = img.min(1)[0].to(dtypes)
        hue[img[:, 2] == value] = (
            4.0
            + ((img[:, 0] - img[:, 1]) / (value - img_min + eps))[img[:, 2] == value]
        )
        hue[img[:, 1] == value] = (
            2.0
            + ((img[:, 2] - img[:, 0]) / (value - img_min + eps))[img[:, 1] == value]
        )
        hue[img[:, 0] == value] = (
            0.0
            + ((img[:, 1] - img[:, 2]) / (value - img_min + eps))[img[:, 0] == value]
        ) % 6

        hue[img.min(1)[0] == value] = 0.0
        hue = hue / 6.0

        saturation = (value - img_min) / (value + eps)
        saturation[value == 0] = 0

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)

        k = self.density_k
        self.this_k = k.item()

        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k)
        cx = (2.0 * pi * hue).cos()
        cy = (2.0 * pi * hue).sin()
        X = color_sensitive * saturation * cx
        Y = color_sensitive * saturation * cy
        Z = value
        xyz = torch.cat([X, Y, Z], dim=1)
        return xyz

    def PHVIT(self, img):
        eps = 1e-8
        H, V, I = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]  # noqa: E741

        # clip
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        I = torch.clamp(I, 0, 1)  # noqa: E741

        v = I
        k = self.this_k
        color_sensitive = ((v * 0.5 * pi).sin() + eps).pow(k)
        H = (H) / (color_sensitive + eps)
        V = (V) / (color_sensitive + eps)
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        h = torch.atan2(V, H) / (2 * pi)
        h = h % 1
        s = torch.sqrt(H**2 + V**2)

        if self.gated:
            s = s * 1.3

        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1.0 - s)
        q = v * (1.0 - (f * s))
        t = v * (1.0 - ((1.0 - f) * s))

        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5

        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]

        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]

        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]

        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]

        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]

        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        if self.gated2:
            rgb = rgb * self.alpha
        return rgb
