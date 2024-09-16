from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from spandrel.util import store_hyperparameters
from spandrel.util.timm import to_2tuple


class LSAB(nn.Module):
    def __init__(self, in_channels=55, num_head=5):
        super().__init__()
        m_body = []
        for _ in range(2):
            m_body.append(SwinT(num_head=num_head, n_feats=in_channels))
        self.body = nn.Sequential(*m_body)

    def forward(self, input):
        out_fused = self.body(input)
        return out_fused


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=padding,
        bias=True,
        dilation=dilation,
        groups=groups,
    )


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(
    in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1
):
    # import pdb
    # pdb.set_trace()
    conv = conv_layer(
        in_channels, out_channels * (upscale_factor**2), kernel_size, stride
    )
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == "reflect":
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == "replicate":
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError(f"padding layer [{pad_type:s}] is not implemented")
    return layer


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == "relu":
        layer = nn.ReLU(inplace)
    elif act_type == "lrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(f"activation layer [{act_type:s}] is not found")
    return layer


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == "batch":
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == "instance":
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError(f"normalization layer [{norm_type:s}] is not found")
    return layer


def conv_block(
    in_nc,
    out_nc,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    pad_type="zero",
    norm_type=None,
    act_type="relu",
):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != "zero" else None
    padding = padding if pad_type == "zero" else 0

    c = nn.Conv2d(
        in_nc,
        out_nc,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        groups=groups,
    )
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


class SwinT(nn.Module):
    def __init__(self, num_head=5, n_feats=55):
        super().__init__()
        m = []
        depth = 2
        num_heads = num_head
        window_size = 16
        resolution = 64
        mlp_ratio = 2.0
        m.append(
            BasicLayer(
                dim=n_feats,
                depth=depth,
                resolution=resolution,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
            )
        )
        self.transformer_body = nn.Sequential(*m)

    def forward(self, x):
        res = self.transformer_body(x)
        return res


class BasicLayer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        resolution,
        depth=2,
        num_heads=8,
        window_size=8,
        mlp_ratio=1.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=None,
    ):
        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.depth = depth
        self.window_size = window_size
        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    resolution=resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                )
                for i in range(depth)
            ]
        )
        self.patch_embed = PatchEmbed(embed_dim=dim, norm_layer=norm_layer)
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        if mod_pad_h != 0 or mod_pad_w != 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        return x, h, w

    def forward(self, x):
        x, h, w = self.check_image_size(x)
        _, _, H, W = x.size()
        x_size = (H, W)
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x, x_size)
        x = self.patch_unembed(x, x_size)
        if h != H or w != W:
            x = x[:, :, 0:h, 0:w].contiguous()
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        resolution,
        num_heads,
        window_size=8,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.dim = dim
        self.resolution = to_2tuple(resolution)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer
        )

    def forward(self, x, x_size):
        H, W = x_size
        B, _L, C = x.shape

        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x
        shifted_x = self.attn(shifted_x, H, W, mask=None)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = x + self.mlp(x)
        return x


class LocalModule(nn.Sequential):
    def __init__(self, channels):
        super().__init__()
        self.add_module("pointwise_prenorm_0", nn.BatchNorm2d(channels))
        self.add_module(
            "pointwise_conv_0", nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        )
        self.add_module(
            "depthwise_conv",
            nn.Conv2d(
                channels,
                channels,
                padding=1,
                kernel_size=3,
                groups=channels,
                bias=False,
            ),
        )
        self.add_module("pointwise_prenorm_1", nn.BatchNorm2d(channels))
        self.add_module(
            "pointwise_conv_1", nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        )


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-2)
        self.local = LocalModule(dim)

    def forward(self, x, H, W, mask=None):
        temp = x.permute(0, 3, 1, 2).contiguous()
        local = self.local(temp)
        local = local + temp
        local = local.permute(0, 2, 3, 1).contiguous()
        qkv = self.qkv(local)
        # partition windows
        qkv = window_partition(
            qkv, self.window_size[0]
        )  # nW*B, window_size, window_size, C

        B_, _, _, C = qkv.shape

        qkv = qkv.view(
            -1, self.window_size[0] * self.window_size[1], C
        )  # nW*B, window_size*window_size, C

        N = self.window_size[0] * self.window_size[1]
        C = C // 3

        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        k = self.softmax(k)

        q = q * self.scale
        attn = k.transpose(-2, -1) @ v
        x = (q @ attn).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)

        # merge windows
        x = x.view(-1, self.window_size[0], self.window_size[1], C)
        x = window_reverse(x, self.window_size[0], H, W)  # B H' W' C
        x = x + local
        return x


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim=50, norm_layer=None):
        super().__init__()

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim=50):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, _HW, _C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


@store_hyperparameters()
class DCTLSA(nn.Module):
    hyperparameters = {}

    def __init__(
        self,
        *,
        in_nc=3,
        nf=55,
        num_modules=6,
        out_nc=3,
        upscale=4,
        num_head=5,
    ):
        super().__init__()
        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)
        # self.dctb = DCTB(nf=nf,num_modules=num_modules)
        # //The DCTB code Start
        self.B1 = LSAB(in_channels=nf, num_head=num_head)
        self.B2 = LSAB(in_channels=nf, num_head=num_head)
        self.B3 = LSAB(in_channels=nf, num_head=num_head)
        self.B4 = LSAB(in_channels=nf, num_head=num_head)
        self.B5 = LSAB(in_channels=nf, num_head=num_head)
        self.B6 = LSAB(in_channels=nf, num_head=num_head)
        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type="lrelu")
        self.c1 = conv_block(nf * 2, nf, kernel_size=1, act_type="lrelu")
        self.c2 = conv_block(nf * 3, nf, kernel_size=1, act_type="lrelu")
        self.c3 = conv_block(nf * 4, nf, kernel_size=1, act_type="lrelu")
        self.c4 = conv_block(nf * 5, nf, kernel_size=1, act_type="lrelu")
        self.c5 = conv_block(nf * 6, nf, kernel_size=1, act_type="lrelu")
        # //The DCTB code End
        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)

    def forward(self, input):
        out_fea = self.fea_conv(input)

        out_B1 = self.B1(out_fea)
        out_B10 = torch.cat([out_fea, out_B1], dim=1)
        out_B11 = self.c1(out_B10)

        out_B2 = self.B2(out_B11)
        out_B20 = torch.cat([out_B10, out_B2], dim=1)
        out_B22 = self.c2(out_B20)

        out_B3 = self.B3(out_B22)
        out_B30 = torch.cat([out_B20, out_B3], dim=1)
        out_B33 = self.c3(out_B30)

        out_B4 = self.B4(out_B33)
        out_B40 = torch.cat([out_B30, out_B4], dim=1)
        out_B44 = self.c4(out_B40)

        out_B5 = self.B5(out_B44)
        out_B50 = torch.cat([out_B40, out_B5], dim=1)
        out_B55 = self.c5(out_B50)

        out_B6 = self.B6(out_B55)

        out_B = self.c(
            torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1)
        )
        # out_B = self.dctb(out_fea)
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output
