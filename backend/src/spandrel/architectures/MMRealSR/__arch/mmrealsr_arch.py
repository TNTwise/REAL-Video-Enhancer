import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules.batchnorm import _BatchNorm  # type: ignore

from spandrel.util import store_hyperparameters


@torch.no_grad()  # type: ignore
def default_init_weights(module_list, scale: float = 1, bias_fill: float = 0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:  # type: ignore
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:  # type: ignore
                    m.bias.data.fill_(bias_fill)


def pixel_unshuffle(x, scale: int):
    """Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1
        )

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


class AffineModulate(nn.Module):
    def __init__(self, degradation_dim=128, num_feat=64):
        super().__init__()
        degradation_dim = 512  # 256 #64
        self.fc = nn.Sequential(
            nn.Linear(degradation_dim, (degradation_dim + num_feat * 2) // 2),
            nn.ReLU(True),
            nn.Linear(
                (degradation_dim + num_feat * 2) // 2,
                (degradation_dim + num_feat * 2) // 2,
            ),
            nn.ReLU(True),
            nn.Linear((degradation_dim + num_feat * 2) // 2, num_feat * 2),
        )
        default_init_weights([self.fc], 0.1)

    def forward(self, x, d):
        d = self.fc(d)
        d = d.view(d.size(0), d.size(1), 1, 1)
        gamma, beta = torch.chunk(d, chunks=2, dim=1)

        return (1 + gamma) * x + beta


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class DEResNet(nn.Module):
    """Degradation Estimator with ResNetNoBN arch. v2.1, no vector anymore

    As shown in paper 'Towards Flexible Blind JPEG Artifacts Removal',
    resnet arch works for image quality estimation.

    Args:
        num_in_ch (int): channel number of inputs. Default: 3.
        num_degradation (int): num of degradation the DE should estimate. Default: 2(blur+noise).
        degradation_embed_size (int): embedding size of each degradation vector.
        degradation_degree_actv (int): activation function for degradation degree scalar. Default: sigmoid.
        num_feats (list): channel number of each stage.
        num_blocks (list): residual block of each stage.
        downscales (list): downscales of each stage.
    """

    def __init__(
        self,
        num_in_ch=3,
        num_degradation=2,
        degradation_degree_actv="sigmoid",
        num_feats=[64, 128, 256, 512],
        num_blocks=[2, 2, 2, 2],
        downscales=[2, 2, 2, 1],
    ):
        super().__init__()

        assert isinstance(num_feats, list)
        assert isinstance(num_blocks, list)
        assert isinstance(downscales, list)
        assert len(num_feats) == len(num_blocks) and len(num_feats) == len(downscales)

        num_stage = len(num_feats)

        self.conv_first = nn.ModuleList()
        for _ in range(num_degradation):
            self.conv_first.append(nn.Conv2d(num_in_ch, num_feats[0], 3, 1, 1))
        self.body = nn.ModuleList()
        for _ in range(num_degradation):
            body = []
            for stage in range(num_stage):
                for _ in range(num_blocks[stage]):
                    body.append(ResidualBlockNoBN(num_feats[stage]))
                if downscales[stage] == 1:
                    if (
                        stage < num_stage - 1
                        and num_feats[stage] != num_feats[stage + 1]
                    ):
                        body.append(
                            nn.Conv2d(num_feats[stage], num_feats[stage + 1], 3, 1, 1)
                        )
                    continue
                elif downscales[stage] == 2:
                    body.append(
                        nn.Conv2d(
                            num_feats[stage],
                            num_feats[min(stage + 1, num_stage - 1)],
                            3,
                            2,
                            1,
                        )
                    )
                else:
                    raise NotImplementedError
            self.body.append(nn.Sequential(*body))

        # self.body = nn.Sequential(*body)

        self.num_degradation = num_degradation
        self.fc_degree = nn.ModuleList()
        if degradation_degree_actv == "sigmoid":
            actv = nn.Sigmoid
        elif degradation_degree_actv == "tanh":
            actv = nn.Tanh
        else:
            raise NotImplementedError(
                f"only sigmoid and tanh are supported for degradation_degree_actv, "
                f"{degradation_degree_actv} is not supported yet."
            )
        for _ in range(num_degradation):
            self.fc_degree.append(
                nn.Sequential(
                    nn.Linear(num_feats[-1], 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 1),
                    actv(),
                )
            )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        default_init_weights([self.conv_first, self.body, self.fc_degree], 0.1)

    def forward(self, x):
        degrees = []
        for i in range(self.num_degradation):
            x_out = self.conv_first[i](x)
            feat = self.body[i](x_out)
            feat = self.avg_pool(feat)
            feat = feat.squeeze(-1).squeeze(-1)
            # for i in range(self.num_degradation):
            degrees.append(self.fc_degree[i](feat).squeeze(-1))

        return degrees


class MMRRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used in ESRGAN. v2.1

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(
        self,
        num_in_ch,
        num_out_ch,
        scale=4,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        de_net_type="DEResNet",
        num_degradation=2,
        degradation_degree_actv="sigmoid",
        num_feats=[64, 128, 256, 512],
        num_blocks=[2, 2, 2, 2],
        downscales=[2, 2, 2, 1],
    ):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.ModuleList()
        # affine modulate list
        self.am_list = nn.ModuleList()
        for _ in range(num_block):
            self.body.append(RRDB(num_feat, num_grow_ch=num_grow_ch))
            self.am_list.append(AffineModulate(degradation_dim=512, num_feat=num_feat))

        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.num_degradation = num_degradation
        self.num_block = num_block

        # degradation net
        assert de_net_type == "DEResNet"
        self.de_net = DEResNet(
            num_in_ch=num_in_ch,
            num_degradation=num_degradation,
            degradation_degree_actv=degradation_degree_actv,
            num_feats=num_feats,
            num_blocks=num_blocks,
            downscales=downscales,
        )

        # degradation degrees embedding
        self.dd_embed = nn.Sequential(
            nn.Linear(num_degradation, 512),  # 512 is real!!!!!
            nn.ReLU(True),
        )
        default_init_weights([self.dd_embed], 0.1)

    def forward(self, x, custom_degrees=(None, None, None), anchor=None):
        b = x.shape[0]
        if anchor is not None:
            b, _n, c, w, h = x.shape
            x = torch.cat([x, anchor], dim=1).contiguous()
            x = x.view(b * 5, c, w, h)
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        feat_res = feat

        # predict the degradation embeddings and degrees
        degrees = self.de_net(x)
        new_degrees = []
        for i in range(self.num_degradation):
            if custom_degrees[i] is None:
                reg = degrees[i].view(b, 5)
                min = torch.zeros_like(reg[:, -2].unsqueeze(-1))
                max = torch.ones_like(reg[:, -2].unsqueeze(-1))
                new_degrees.append(torch.cat([reg[:, :-2], min, max], dim=-1).view(-1))
                # print(degrees[i].shape)
            else:
                new_degrees.append(
                    torch.zeros_like(degrees[i]).fill_(custom_degrees[i])
                )

        concat_degrees = torch.stack(new_degrees, dim=1)
        d_embedding = self.dd_embed(concat_degrees)

        for i in range(self.num_block):
            feat = self.body[i](feat)
            feat = self.am_list[i](feat, d_embedding)
        feat = self.conv_body(feat)
        feat = feat_res + feat
        # upsample
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out, degrees


class MMRRDBNet_decouple(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used in ESRGAN. v2.1

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(
        self,
        num_in_ch,
        num_out_ch,
        scale=4,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        de_net_type="DEResNet",
        num_degradation=2,
        degradation_degree_actv="sigmoid",
        num_feats=[64, 128, 256, 512],
        num_blocks=[2, 2, 2, 2],
        downscales=[2, 2, 2, 1],
    ):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.ModuleList()
        # affine modulate list
        self.am_list = nn.ModuleList()
        for _ in range(num_block):
            self.body.append(RRDB(num_feat, num_grow_ch=num_grow_ch))
            self.am_list.append(AffineModulate(degradation_dim=512, num_feat=num_feat))

        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.num_degradation = num_degradation
        self.num_block = num_block

        # degradation net
        assert de_net_type == "DEResNet"
        self.de_net = DEResNet(
            num_in_ch=num_in_ch,
            num_degradation=num_degradation,
            degradation_degree_actv=degradation_degree_actv,
            num_feats=num_feats,
            num_blocks=num_blocks,
            downscales=downscales,
        )

        # degradation degrees embedding
        self.dd_embed = nn.Sequential(
            nn.Linear(num_degradation, 512),  # 512 is reall
            nn.ReLU(True),
        )
        default_init_weights([self.dd_embed], 0.1)

    def forward(self, x, custom_degrees=(None, None, None), anchor=None):
        b = x.shape[0]
        if anchor is not None:
            b, _n, c, w, h = x.shape
            x = torch.cat([x, anchor], dim=1).contiguous()
            x = x.view(b * 5, c, w, h)
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        feat_res = feat

        # predict the degradation embeddings and degrees
        with torch.no_grad():
            degrees = self.de_net(x)
        new_degrees = []
        for i in range(self.num_degradation):
            if custom_degrees[i] is None:
                reg = degrees[i].view(b, 5)
                min = torch.zeros_like(reg[:, -2].unsqueeze(-1))
                max = torch.ones_like(reg[:, -2].unsqueeze(-1))
                new_degrees.append(
                    torch.cat([reg[:, :-2], min, max], dim=-1).view(-1).detach()
                )
                # print(degrees[i].shape)
            else:
                new_degrees.append(
                    torch.zeros_like(degrees[i]).fill_(custom_degrees[i])
                )

        concat_degrees = torch.stack(new_degrees, dim=1)
        d_embedding = self.dd_embed(concat_degrees)

        for i in range(self.num_block):
            feat = self.body[i](feat)
            feat = self.am_list[i](feat, d_embedding)
        feat = self.conv_body(feat)
        feat = feat_res + feat
        # upsample
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out, degrees


@store_hyperparameters()
class MMRRDBNet_test(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used in ESRGAN. v2.1

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    hyperparameters = {}

    def __init__(
        self,
        *,
        num_in_ch,
        num_out_ch,
        scale=4,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        de_net_type="DEResNet",
        num_degradation=2,
        degradation_degree_actv="sigmoid",
        num_feats=[64, 128, 256, 512],
        num_blocks=[2, 2, 2, 2],
        downscales=[2, 2, 2, 1],
    ):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.ModuleList()
        # affine modulate list
        self.am_list = nn.ModuleList()
        for _ in range(num_block):
            self.body.append(RRDB(num_feat, num_grow_ch=num_grow_ch))
            self.am_list.append(AffineModulate(degradation_dim=512, num_feat=num_feat))

        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.num_degradation = num_degradation
        self.num_block = num_block

        # degradation net
        assert de_net_type == "DEResNet"
        self.de_net = DEResNet(
            num_in_ch=num_in_ch,
            num_degradation=num_degradation,
            degradation_degree_actv=degradation_degree_actv,
            num_feats=num_feats,
            num_blocks=num_blocks,
            downscales=downscales,
        )

        # degradation degrees embedding
        self.dd_embed = nn.Sequential(
            nn.Linear(num_degradation, 512),  # 512 is real !!!!!
            nn.ReLU(True),
        )
        default_init_weights([self.dd_embed], 0.1)

    def forward(self, x, custom_degrees=(None, None)):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        feat_res = feat

        # predict the degradation embeddings and degrees
        degrees = self.de_net(x)
        new_degrees = []
        for i in range(self.num_degradation):
            if custom_degrees[i] is None:
                new_degrees.append(degrees[i])
            else:
                new_degrees.append(
                    torch.zeros_like(degrees[i]).fill_(custom_degrees[i])
                )

        concat_degrees = torch.stack(new_degrees, dim=1)
        d_embedding = self.dd_embed(concat_degrees)

        for i in range(self.num_block):
            feat = self.body[i](feat)
            feat = self.am_list[i](feat, d_embedding)

        feat = self.conv_body(feat)
        feat = feat_res + feat
        # upsample
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))

        return out, degrees
