import torch
from torch import nn
from torch.nn.functional import interpolate
from torch.nn import init
from torch.nn.modules.batchnorm import _BatchNorm

"""
Originally from Tencent ARC, Modified by HolyWu
"""
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
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
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


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
        super(ResidualBlockNoBN, self).__init__()
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


class MyPixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        b, c, hh, hw = x.size()
        out_channel = c // (self.upscale_factor**2)
        h = hh * self.upscale_factor
        w = hw * self.upscale_factor
        x_view = x.view(b, out_channel, self.upscale_factor, self.upscale_factor, hh, hw)
        return x_view.permute(0, 1, 4, 2, 5, 3).reshape(b, out_channel, h, w)


class MyPixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        b, c, hh, hw = x.size()
        out_channel = c * (self.downscale_factor**2)
        h = hh // self.downscale_factor
        w = hw // self.downscale_factor
        x_view = x.view(b, c, h, self.downscale_factor, w, self.downscale_factor)
        return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class RightAlignMSConvResidualBlocks(nn.Module):
    """right align multi-scale ConvResidualBlocks, currently only support 3 scales (1, 2, 4)"""

    def __init__(self, num_in_ch=3, num_state_ch=64, num_out_ch=64, num_block=(5, 3, 2)):
        super().__init__()

        assert len(num_block) == 3
        assert num_block[0] >= num_block[1] >= num_block[2]
        self.num_block = num_block

        self.conv_s1_first = nn.Sequential(
            nn.Conv2d(num_in_ch, num_state_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.conv_s2_first = nn.Sequential(
            nn.Conv2d(num_state_ch, num_state_ch, 3, 2, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.conv_s4_first = nn.Sequential(
            nn.Conv2d(num_state_ch, num_state_ch, 3, 2, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.body_s1_first = nn.ModuleList()
        for _ in range(num_block[0]):
            self.body_s1_first.append(ResidualBlockNoBN(num_feat=num_state_ch))
        self.body_s2_first = nn.ModuleList()
        for _ in range(num_block[1]):
            self.body_s2_first.append(ResidualBlockNoBN(num_feat=num_state_ch))
        self.body_s4_first = nn.ModuleList()
        for _ in range(num_block[2]):
            self.body_s4_first.append(ResidualBlockNoBN(num_feat=num_state_ch))

        self.fusion = nn.Sequential(
            nn.Conv2d(3 * num_state_ch, 2 * num_out_ch, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(2 * num_out_ch, num_out_ch, 3, 1, 1, bias=True),
        )

    def up(self, x, scale=2):
        return interpolate(x, scale_factor=scale, mode='bilinear')

    def forward(self, x):
        x_s1 = self.conv_s1_first(x)
        x_s2 = self.conv_s2_first(x_s1)
        x_s4 = self.conv_s4_first(x_s2)

        flag_s2 = False
        flag_s4 = False
        for i in range(0, self.num_block[0]):
            x_s1 = self.body_s1_first[i](
                x_s1 + (self.up(x_s2, 2) if flag_s2 else 0) + (self.up(x_s4, 4) if flag_s4 else 0))
            if i >= self.num_block[0] - self.num_block[1]:
                x_s2 = self.body_s2_first[i - self.num_block[0] + self.num_block[1]](
                    x_s2 + (self.up(x_s4, 2) if flag_s4 else 0))
                flag_s2 = True
            if i >= self.num_block[0] - self.num_block[2]:
                x_s4 = self.body_s4_first[i - self.num_block[0] + self.num_block[2]](x_s4)
                flag_s4 = True

        x_fusion = self.fusion(torch.cat((x_s1, self.up(x_s2, 2), self.up(x_s4, 4)), dim=1))

        return x_fusion


class AnimeSR(nn.Module):
    """
    Multi-Scale, unidirectional Recurrent, Sliding Window (MSRSW)
    The implementation refers to paper: Efficient Video Super-Resolution through Recurrent Latent Space Propagation
    """

    def __init__(self, num_feat=64, num_block=(5, 3, 2), netscale=4):
        super(AnimeSR, self).__init__()
        self.num_feat = num_feat

        # 3(img channel) * 3(prev cur nxt 3 imgs) + 3(hr img channel) * netscale * netscale + num_feat
        self.recurrent_cell = RightAlignMSConvResidualBlocks(3 * 3 + 3 * netscale * netscale + num_feat, num_feat,
                                                             num_feat + 3 * netscale * netscale, num_block)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.pixel_shuffle = MyPixelShuffle(netscale)
        self.pixel_unshuffle = MyPixelUnshuffle(netscale)
        self.netscale = netscale

    def forward(self, x, fb, state):
        res = x[:, 3:6]
        # pre frame, cur frame, nxt frame, pre sr frame, pre hidden state
        inp = torch.cat((x, self.pixel_unshuffle(fb), state), dim=1)
        # the out contains both state and sr frame
        out = self.recurrent_cell(inp)
        out_img = self.pixel_shuffle(out[:, :3 * self.netscale * self.netscale]) + interpolate(
            res, scale_factor=self.netscale, mode='bilinear')
        out_state = self.lrelu(out[:, 3 * self.netscale * self.netscale:])

        return out_img, out_state