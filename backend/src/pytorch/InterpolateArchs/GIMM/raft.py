# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ginr-ipc: https://github.com/kakaobrain/ginr-ipc
# --------------------------------------------------------

import torch
import torch.nn as nn
import math
import einops
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "xpu" if torch.xpu.is_available() else "cpu")
backwarp_tenGrid = {}


def warp(tenInput, tenFlow):
    origdtype = tenInput.dtype
    tenFlow = tenFlow.float()
    tenInput = tenInput.float()
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device)
            .view(1, 1, 1, tenFlow.shape[3])
            .expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        ).float()
        tenVertical = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device)
            .view(1, 1, tenFlow.shape[2], 1)
            .expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        ).float()
        backwarp_tenGrid[k] = (
            torch.cat([tenHorizontal, tenVertical], 1).to(device).float()
        )

    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
        ],
        1,
    ).float()

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1).float()
    pd = 'border'
    if tenInput.device.type == "mps":
        pd = 'zeros'
        g = g.clamp(-1, 1)
    return torch.nn.functional.grid_sample(
        input=tenInput,
        grid=g,
        mode="bilinear",
        padding_mode=pd,
        align_corners=True,
    ).to(dtype=origdtype)


def normalize_flow(flows):
    # FIXME: MULTI-DIMENSION
    flow_scaler = torch.max(torch.abs(flows).flatten(1), dim=-1)[0].reshape(
        -1, 1, 1, 1, 1
    )
    flows = flows / flow_scaler  # [-1,1]
    # # Adapt to [0,1]
    flows = (flows + 1.0) / 2.0
    return flows, flow_scaler


def unnormalize_flow(flows, flow_scaler):
    return (flows * 2.0 - 1.0) * flow_scaler


def resize(x, scale_factor):
    return F.interpolate(
        x, scale_factor=scale_factor, mode="bilinear", align_corners=False
    )


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0)
    return coords[None].repeat(batch, 1, 1, 1)


def build_coord(img):
    N, C, H, W = img.shape
    coords = coords_grid(N, H // 8, W // 8)
    return coords


def initialize_params(params, init_type, **kwargs):
    fan_in, fan_out = params.shape[0], params.shape[1]
    if init_type is None or init_type == "normal":
        nn.init.normal_(params)
    elif init_type == "kaiming_uniform":
        nn.init.kaiming_uniform_(params, a=math.sqrt(5))
    elif init_type == "uniform_fan_in":
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(params, -bound, bound)
    elif init_type == "zero":
        nn.init.zeros_(params)
    elif "siren" == init_type:
        assert "siren_w0" in kwargs.keys() and "is_first" in kwargs.keys()
        w0 = kwargs["siren_w0"]
        if kwargs["is_first"]:
            w_std = 1 / fan_in
        else:
            w_std = math.sqrt(6.0 / fan_in) / w0
        nn.init.uniform_(params, -w_std, w_std)
    else:
        raise NotImplementedError


def create_params_with_init(
    shape, init_type="normal", include_bias=False, bias_init_type="zero", **kwargs
):
    if not include_bias:
        params = torch.empty([shape[0], shape[1]])
        initialize_params(params, init_type, **kwargs)
        return params
    else:
        params = torch.empty([shape[0] - 1, shape[1]])
        bias = torch.empty([1, shape[1]])

        initialize_params(params, init_type, **kwargs)
        initialize_params(bias, bias_init_type, **kwargs)
        return torch.cat([params, bias], dim=0)


class CoordSampler3D(nn.Module):
    def __init__(self, coord_range, t_coord_only=False):
        super().__init__()
        self.coord_range = coord_range
        self.t_coord_only = t_coord_only

    def shape2coordinate(
        self,
        batch_size,
        spatial_shape,
        t_ids,
        coord_range=(-1.0, 1.0),
        upsample_ratio=1,
        device=None,
    ):
        coords = []
        assert isinstance(t_ids, list)
        _coords = torch.tensor(t_ids, device=device) / 1.0
        coords.append(_coords)
        for num_s in spatial_shape:
            num_s = int(num_s * upsample_ratio)
            _coords = (0.5 + torch.arange(num_s, device=device)) / num_s
            _coords = coord_range[0] + (coord_range[1] - coord_range[0]) * _coords
            coords.append(_coords)
        coords = torch.meshgrid(*coords, indexing="ij")
        coords = torch.stack(coords, dim=-1)
        ones_like_shape = (1,) * coords.ndim
        coords = coords.unsqueeze(0).repeat(batch_size, *ones_like_shape)
        return coords  # (B,T,H,W,3)

    def batchshape2coordinate(
        self,
        batch_size,
        spatial_shape,
        t_ids,
        coord_range=(-1.0, 1.0),
        upsample_ratio=1,
        device=None,
    ):
        coords = []
        _coords = torch.tensor(1, device=device)
        coords.append(_coords)
        for num_s in spatial_shape:
            num_s = int(num_s * upsample_ratio)
            _coords = (0.5 + torch.arange(num_s, device=device)) / num_s
            _coords = coord_range[0] + (coord_range[1] - coord_range[0]) * _coords
            coords.append(_coords)
        coords = torch.meshgrid(*coords, indexing="ij")
        coords = torch.stack(coords, dim=-1)
        ones_like_shape = (1,) * coords.ndim
        # Now coords b,1,h,w,3, coords[...,0]=1.
        coords = coords.unsqueeze(0).repeat(batch_size, *ones_like_shape)
        # assign per-sample timestep within the batch
        coords[..., :1] = coords[..., :1] * t_ids.reshape(-1, 1, 1, 1, 1)
        return coords

    def forward(
        self,
        batch_size,
        s_shape,
        t_ids,
        coord_range=None,
        upsample_ratio=1.0,
        device=None,
    ):
        coord_range = self.coord_range if coord_range is None else coord_range
        if isinstance(t_ids, list):
            coords = self.shape2coordinate(
                batch_size, s_shape, t_ids, coord_range, upsample_ratio, device
            )
        elif isinstance(t_ids, torch.Tensor):
            coords = self.batchshape2coordinate(
                batch_size, s_shape, t_ids, coord_range, upsample_ratio, device
            )
        if self.t_coord_only:
            coords = coords[..., :1]
        return coords


# define siren layer & Siren model
class Sine(nn.Module):
    """Sine activation with scaling.

    Args:
        w0 (float): Omega_0 parameter from SIREN paper.
    """

    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class HypoNet(nn.Module):
    r"""
    The Hyponetwork with a coordinate-based MLP to be modulated.
    """

    def __init__(self, add_coord_dim=32):
        super().__init__()
        self.use_bias = True
        self.num_layer = 5
        self.hidden_dims = [128]
        self.add_coord_dim = add_coord_dim

        if len(self.hidden_dims) == 1:
            self.hidden_dims = self.hidden_dims * (
                self.num_layer - 1
            )  # exclude output layer
        else:
            assert len(self.hidden_dims) == self.num_layer - 1

        # after computes the shape of trainable parameters, initialize them
        self.params_dict = None
        self.params_shape_dict = self.compute_params_shape()
        self.activation = Sine(1.0)
        self.build_base_params_dict()
        self.output_bias = 0.5

        self.normalize_weight = True

        self.ignore_base_param_dict = {name: False for name in self.params_dict}

    @staticmethod
    def subsample_coords(coords, subcoord_idx=None):
        if subcoord_idx is None:
            return coords

        batch_size = coords.shape[0]
        sub_coords = []
        coords = coords.view(batch_size, -1, coords.shape[-1])
        for idx in range(batch_size):
            sub_coords.append(coords[idx : idx + 1, subcoord_idx[idx]])
        sub_coords = torch.cat(sub_coords, dim=0)
        return sub_coords

    def forward(self, coord, modulation_params_dict=None, pixel_latent=None):
        origdtype = coord[0].dtype
        sub_idx = None
        if isinstance(coord, tuple):
            coord, sub_idx = coord[0], coord[1]

        if modulation_params_dict is not None:
            self.check_valid_param_keys(modulation_params_dict)

        batch_size, coord_shape, input_dim = (
            coord.shape[0],
            coord.shape[1:-1],
            coord.shape[-1],
        )
        coord = coord.view(batch_size, -1, input_dim)  # flatten the coordinates
        assert pixel_latent is not None
        pixel_latent = F.interpolate(
            pixel_latent.permute(0, 3, 1, 2),
            size=(coord_shape[1], coord_shape[2]),
            mode="bilinear",
        ).permute(0, 2, 3, 1)
        pixel_latent_dim = pixel_latent.shape[-1]
        pixel_latent = pixel_latent.view(batch_size, -1, pixel_latent_dim)
        hidden = coord

        hidden = torch.cat([pixel_latent, hidden], dim=-1)

        hidden = self.subsample_coords(hidden, sub_idx)

        for idx in range(5):
            param_key = f"linear_wb{idx}"
            base_param = einops.repeat(
                self.params_dict[param_key], "n m -> b n m", b=batch_size
            )

            if (modulation_params_dict is not None) and (
                param_key in modulation_params_dict.keys()
            ):
                modulation_param = modulation_params_dict[param_key]
            else:
                modulation_param = torch.ones_like(base_param[:, :-1])

            ones = torch.ones(*hidden.shape[:-1], 1, device=hidden.device)
            hidden = torch.cat([hidden, ones], dim=-1).to(dtype=origdtype)

            base_param_w, base_param_b = (
                base_param[:, :-1, :],
                base_param[:, -1:, :],
            )

            if self.ignore_base_param_dict[param_key]:
                base_param_w = 1.0
            param_w = base_param_w * modulation_param
            if self.normalize_weight:
                param_w = F.normalize(param_w, dim=1)
            modulated_param = torch.cat([param_w, base_param_b], dim=1)

            # print([param_key,hidden.shape,modulated_param.shape])
            hidden = torch.bmm(hidden, modulated_param)

            if idx < (5 - 1):
                hidden = self.activation(hidden)

        outputs = hidden + self.output_bias
        if sub_idx is None:
            outputs = outputs.view(batch_size, *coord_shape, -1)
        return outputs

    def compute_params_shape(self):
        """
        Computes the shape of MLP parameters.
        The computed shapes are used to build the initial weights by `build_base_params_dict`.
        """
        use_bias = self.use_bias

        param_shape_dict = dict()

        fan_in = 3
        add_dim = self.add_coord_dim
        fan_in = fan_in + add_dim
        fan_in = fan_in + 1 if use_bias else fan_in

        for i in range(4):
            fan_out = self.hidden_dims[i]
            param_shape_dict[f"linear_wb{i}"] = (fan_in, fan_out)
            fan_in = fan_out + 1 if use_bias else fan_out

        param_shape_dict[f"linear_wb{4}"] = (fan_in, 2)
        return param_shape_dict

    def build_base_params_dict(self):
        assert self.params_shape_dict
        params_dict = nn.ParameterDict()
        for idx, (name, shape) in enumerate(self.params_shape_dict.items()):
            is_first = idx == 0
            params = create_params_with_init(
                shape,
                init_type="siren",
                include_bias=self.use_bias,
                bias_init_type="siren",
                is_first=is_first,
                siren_w0=1.0,  # valid only for siren
            )
            params = nn.Parameter(params)
            params_dict[name] = params
        self.set_params_dict(params_dict)

    def check_valid_param_keys(self, params_dict):
        predefined_params_keys = self.params_shape_dict.keys()
        for param_key in params_dict.keys():
            if param_key in predefined_params_keys:
                continue
            else:
                raise KeyError

    def set_params_dict(self, params_dict):
        self.check_valid_param_keys(params_dict)
        self.params_dict = params_dict


class LateralBlock(nn.Module):
    def __init__(self, dim):
        super(LateralBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        res = x
        x = self.layers(x)
        return x + res


def convrelu(
    in_channels,
    out_channels,
    kernel_size=3,
    stride=1,
    padding=1,
    dilation=1,
    groups=1,
    bias=True,
):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias,
        ),
        nn.PReLU(out_channels),
    )


def multi_flow_combine(
    comb_block, img0, img1, flow0, flow1, mask=None, img_res=None, mean=None
):
    assert mean is None
    b, c, h, w = flow0.shape
    num_flows = c // 2
    flow0 = flow0.reshape(b, num_flows, 2, h, w).reshape(-1, 2, h, w)
    flow1 = flow1.reshape(b, num_flows, 2, h, w).reshape(-1, 2, h, w)

    mask = (
        mask.reshape(b, num_flows, 1, h, w).reshape(-1, 1, h, w)
        if mask is not None
        else None
    )
    img_res = (
        img_res.reshape(b, num_flows, 3, h, w).reshape(-1, 3, h, w)
        if img_res is not None
        else 0
    )
    img0 = torch.stack([img0] * num_flows, 1).reshape(-1, 3, h, w)
    img1 = torch.stack([img1] * num_flows, 1).reshape(-1, 3, h, w)
    mean = (
        torch.stack([mean] * num_flows, 1).reshape(-1, 1, 1, 1)
        if mean is not None
        else 0
    )

    img0_warp = warp(img0, flow0)
    img1_warp = warp(img1, flow1)
    img_warps = mask * img0_warp + (1 - mask) * img1_warp + mean + img_res
    img_warps = img_warps.reshape(b, num_flows, 3, h, w)

    res = comb_block(img_warps.view(b, -1, h, w))
    imgt_pred = img_warps.mean(1) + res

    imgt_pred = (imgt_pred + 1.0) / 2

    return imgt_pred


class ResBlock(nn.Module):
    def __init__(self, in_channels, side_channels, bias=True):
        super(ResBlock, self).__init__()
        self.side_channels = side_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias
            ),
            nn.PReLU(in_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                side_channels,
                side_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
            nn.PReLU(side_channels),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias
            ),
            nn.PReLU(in_channels),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                side_channels,
                side_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
            nn.PReLU(side_channels),
        )
        self.conv5 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.conv1(x)

        res_feat = out[:, : -self.side_channels, ...]
        side_feat = out[:, -self.side_channels :, :, :]
        side_feat = self.conv2(side_feat)
        out = self.conv3(torch.cat([res_feat, side_feat], 1))

        res_feat = out[:, : -self.side_channels, ...]
        side_feat = out[:, -self.side_channels :, :, :]
        side_feat = self.conv4(side_feat)
        out = self.conv5(torch.cat([res_feat, side_feat], 1))

        out = self.prelu(x + out)
        return out


class BasicUpdateBlock(nn.Module):
    def __init__(
        self,
        cdim,
        hidden_dim,
        flow_dim,
        corr_dim,
        corr_dim2,
        fc_dim,
        corr_levels=4,
        radius=3,
        scale_factor=None,
        out_num=1,
    ):
        super(BasicUpdateBlock, self).__init__()
        cor_planes = corr_levels * (2 * radius + 1) ** 2

        self.scale_factor = scale_factor
        self.convc1 = nn.Conv2d(2 * cor_planes, corr_dim, 1, padding=0)
        self.convc2 = nn.Conv2d(corr_dim, corr_dim2, 3, padding=1)
        self.convf1 = nn.Conv2d(4, flow_dim * 2, 7, padding=3)
        self.convf2 = nn.Conv2d(flow_dim * 2, flow_dim, 3, padding=1)
        self.conv = nn.Conv2d(flow_dim + corr_dim2, fc_dim, 3, padding=1)

        self.gru = nn.Sequential(
            nn.Conv2d(fc_dim + 4 + cdim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )

        self.feat_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(hidden_dim, cdim, 3, padding=1),
        )

        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(hidden_dim, 4 * out_num, 3, padding=1),
        )

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, net, flow, corr):
        net = (
            resize(net, 1 / self.scale_factor) if self.scale_factor is not None else net
        )
        cor = self.lrelu(self.convc1(corr))
        cor = self.lrelu(self.convc2(cor))
        flo = self.lrelu(self.convf1(flow))
        flo = self.lrelu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        inp = self.lrelu(self.conv(cor_flo))
        inp = torch.cat([inp, flow, net], dim=1)

        out = self.gru(inp)
        delta_net = self.feat_head(out)
        delta_flow = self.flow_head(out)

        if self.scale_factor is not None:
            delta_net = resize(delta_net, scale_factor=self.scale_factor)
            delta_flow = self.scale_factor * resize(
                delta_flow, scale_factor=self.scale_factor
            )
        return delta_net, delta_flow


def get_bn():
    return nn.BatchNorm2d


class NewInitDecoder(nn.Module):
    def __init__(self, in_ch, skip_ch):
        super().__init__()
        norm_layer = get_bn()

        self.upsample = nn.Sequential(
            nn.PixelShuffle(2),
            convrelu(in_ch // 4, in_ch // 4, 5, 1, 2),
            convrelu(in_ch // 4, in_ch // 4),
            convrelu(in_ch // 4, in_ch // 4),
            convrelu(in_ch // 4, in_ch // 4),
            convrelu(in_ch // 4, in_ch // 2),
            nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=1),
            norm_layer(in_ch // 2),
            nn.ReLU(inplace=True),
        )

        in_ch = in_ch // 2
        self.convblock = nn.Sequential(
            convrelu(in_ch * 2 + 16, in_ch, kernel_size=1, padding=0),
            ResBlock(in_ch, skip_ch),
            ResBlock(in_ch, skip_ch),
            ResBlock(in_ch, skip_ch),
            nn.Conv2d(in_ch, in_ch + 5, 3, 1, 1, 1, 1, True),
        )

    def forward(self, f0, f1, flow0_in, flow1_in, img0=None, img1=None):
        f0 = self.upsample(f0)
        f1 = self.upsample(f1)
        f0_warp_ks = warp(f0, flow0_in)
        f1_warp_ks = warp(f1, flow1_in)

        f_in = torch.cat([f0_warp_ks, f1_warp_ks, flow0_in, flow1_in], dim=1)

        assert img0 is not None
        assert img1 is not None
        scale_factor = f_in.shape[2] / img0.shape[2]
        img0 = resize(img0, scale_factor=scale_factor)
        img1 = resize(img1, scale_factor=scale_factor)
        warped_img0 = warp(img0, flow0_in)
        warped_img1 = warp(img1, flow1_in)
        f_in = torch.cat([f_in, img0, img1, warped_img0, warped_img1], dim=1)

        out = self.convblock(f_in)
        ft_ = out[:, 4:, ...]
        flow0 = flow0_in + out[:, :2, ...]
        flow1 = flow1_in + out[:, 2:4, ...]
        return flow0, flow1, ft_


class NewMultiFlowDecoder(nn.Module):
    def __init__(self, in_ch, skip_ch, num_flows=3):
        super(NewMultiFlowDecoder, self).__init__()
        norm_layer = get_bn()

        self.upsample = nn.Sequential(
            nn.PixelShuffle(2),
            nn.PixelShuffle(2),
            convrelu(in_ch // (4 * 4), in_ch // 4, 5, 1, 2),
            convrelu(in_ch // 4, in_ch // 4),
            convrelu(in_ch // 4, in_ch // 4),
            convrelu(in_ch // 4, in_ch // 4),
            convrelu(in_ch // 4, in_ch // 2),
            nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=1),
            norm_layer(in_ch // 2),
            nn.ReLU(inplace=True),
        )

        self.num_flows = num_flows
        ch_factor = 2
        self.convblock = nn.Sequential(
            convrelu(in_ch * ch_factor + 17, in_ch * ch_factor),
            ResBlock(in_ch * ch_factor, skip_ch),
            ResBlock(in_ch * ch_factor, skip_ch),
            ResBlock(in_ch * ch_factor, skip_ch),
            nn.Conv2d(in_ch * ch_factor, 8 * num_flows, kernel_size=3, padding=1),
        )

    def forward(self, ft_, f0, f1, flow0, flow1, mask=None, img0=None, img1=None):
        f0 = self.upsample(f0)
        # print([f1.shape,f0.shape])
        f1 = self.upsample(f1)
        n = self.num_flows
        flow0 = 4.0 * resize(flow0, scale_factor=4.0)
        flow1 = 4.0 * resize(flow1, scale_factor=4.0)

        ft_ = resize(ft_, scale_factor=4.0)
        mask = resize(mask, scale_factor=4.0)
        f0_warp = warp(f0, flow0)
        f1_warp = warp(f1, flow1)

        f_in = torch.cat([ft_, f0_warp, f1_warp, flow0, flow1], 1)

        assert mask is not None
        f_in = torch.cat([f_in, mask], 1)

        assert img0 is not None
        assert img1 is not None
        warped_img0 = warp(img0, flow0)
        warped_img1 = warp(img1, flow1)
        f_in = torch.cat([f_in, img0, img1, warped_img0, warped_img1], dim=1)

        out = self.convblock(f_in)
        delta_flow0, delta_flow1, delta_mask, img_res = torch.split(
            out, [2 * n, 2 * n, n, 3 * n], 1
        )
        mask = delta_mask + mask.repeat(1, self.num_flows, 1, 1)
        mask = torch.sigmoid(mask)
        flow0 = delta_flow0 + flow0.repeat(1, self.num_flows, 1, 1)
        flow1 = delta_flow1 + flow1.repeat(1, self.num_flows, 1, 1)

        return flow0, flow1, mask, img_res


def multi_flow_combine(
    comb_block, img0, img1, flow0, flow1, mask=None, img_res=None, mean=None
):
    assert mean is None
    b, c, h, w = flow0.shape
    num_flows = c // 2
    flow0 = flow0.reshape(b, num_flows, 2, h, w).reshape(-1, 2, h, w)
    flow1 = flow1.reshape(b, num_flows, 2, h, w).reshape(-1, 2, h, w)

    mask = (
        mask.reshape(b, num_flows, 1, h, w).reshape(-1, 1, h, w)
        if mask is not None
        else None
    )
    img_res = (
        img_res.reshape(b, num_flows, 3, h, w).reshape(-1, 3, h, w)
        if img_res is not None
        else 0
    )
    img0 = torch.stack([img0] * num_flows, 1).reshape(-1, 3, h, w)
    img1 = torch.stack([img1] * num_flows, 1).reshape(-1, 3, h, w)
    mean = (
        torch.stack([mean] * num_flows, 1).reshape(-1, 1, 1, 1)
        if mean is not None
        else 0
    )

    img0_warp = warp(img0, flow0)
    img1_warp = warp(img1, flow1)
    img_warps = mask * img0_warp + (1 - mask) * img1_warp + mean + img_res
    img_warps = img_warps.reshape(b, num_flows, 3, h, w)

    res = comb_block(img_warps.view(b, -1, h, w))
    imgt_pred = img_warps.mean(1) + res

    imgt_pred = (imgt_pred + 1.0) / 2

    return imgt_pred