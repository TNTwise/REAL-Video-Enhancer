# type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        _N, C, _H, _W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        # print('mu, var', mu.mean(), var.mean())
        # d.append([mu.mean(), var.mean()])
        y = (x - mu) / (var + eps).sqrt()
        weight, bias, y = (
            weight.contiguous(),
            bias.contiguous(),
            y.contiguous(),
        )  # avoid cuda error
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        _N, C, _H, _W = grad_output.size()
        # y, var, weight = ctx.saved_variables
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6, requires_grad=True):
        super().__init__()
        self.register_parameter(
            "weight", nn.Parameter(torch.ones(channels), requires_grad=requires_grad)
        )
        self.register_parameter(
            "bias", nn.Parameter(torch.zeros(channels), requires_grad=requires_grad)
        )
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class KBAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, att, selfk, selfg, selfb, selfw):
        B, nset, H, W = att.shape
        KK = selfk**2
        selfc = x.shape[1]

        att = att.reshape(B, nset, H * W).transpose(-2, -1)

        ctx.selfk, ctx.selfg, ctx.selfc, ctx.KK, ctx.nset = (
            selfk,
            selfg,
            selfc,
            KK,
            nset,
        )
        ctx.x, ctx.att, ctx.selfb, ctx.selfw = x, att, selfb, selfw

        bias = att @ selfb
        attk = att @ selfw

        uf = torch.nn.functional.unfold(x, kernel_size=selfk, padding=selfk // 2)

        # for unfold att / less memory cost
        uf = uf.reshape(B, selfg, selfc // selfg * KK, H * W).permute(0, 3, 1, 2)
        attk = attk.reshape(B, H * W, selfg, selfc // selfg, selfc // selfg * KK)

        x = attk @ uf.unsqueeze(-1)  #
        del attk, uf
        x = x.squeeze(-1).reshape(B, H * W, selfc) + bias
        x = x.transpose(-1, -2).reshape(B, selfc, H, W)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, att, selfb, selfw = ctx.x, ctx.att, ctx.selfb, ctx.selfw
        selfk, selfg, selfc, KK, nset = (
            ctx.selfk,
            ctx.selfg,
            ctx.selfc,
            ctx.KK,
            ctx.nset,
        )

        B, selfc, H, W = grad_output.size()

        dbias = grad_output.reshape(B, selfc, H * W).transpose(-1, -2)

        dselfb = att.transpose(-2, -1) @ dbias
        datt = dbias @ selfb.transpose(-2, -1)

        attk = att @ selfw
        uf = F.unfold(x, kernel_size=selfk, padding=selfk // 2)
        # for unfold att / less memory cost
        uf = uf.reshape(B, selfg, selfc // selfg * KK, H * W).permute(0, 3, 1, 2)
        attk = attk.reshape(B, H * W, selfg, selfc // selfg, selfc // selfg * KK)

        dx = dbias.view(B, H * W, selfg, selfc // selfg, 1)

        dattk = dx @ uf.view(B, H * W, selfg, 1, selfc // selfg * KK)
        duf = attk.transpose(-2, -1) @ dx
        del attk, uf

        dattk = dattk.view(B, H * W, -1)
        datt += dattk @ selfw.transpose(-2, -1)
        dselfw = att.transpose(-2, -1) @ dattk

        duf = duf.permute(0, 2, 3, 4, 1).view(B, -1, H * W)
        dx = F.fold(duf, output_size=(H, W), kernel_size=selfk, padding=selfk // 2)

        datt = datt.transpose(-1, -2).view(B, nset, H, W)

        return dx, datt, None, None, dselfb, dselfw
