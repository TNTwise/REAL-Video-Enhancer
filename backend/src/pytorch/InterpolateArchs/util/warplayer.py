import torch
import torch.nn.functional as F


def warp(tenInput, tenFlow, tenFlow_div, backwarp_tenGrid):
    dtype = tenInput.dtype
    tenInput = tenInput.to(torch.float)
    tenFlow = tenFlow.to(torch.float)

    tenFlow = torch.cat(
        [tenFlow[:, 0:1] / tenFlow_div[0], tenFlow[:, 1:2] / tenFlow_div[1]], 1
    )
    g = (backwarp_tenGrid + tenFlow).permute(0, 2, 3, 1)
    pd = "border"
    if tenInput.device.type == "mps":
        pd = "zeros"
        g = g.clamp(-1, 1)
        return F.grid_sample(
            input=tenInput,
            grid=g,
            mode="bilinear",
            padding_mode=pd,
            align_corners=True,
        ).to(dtype)
    else:
        return torch.ops.aten.grid_sampler_2d(tenInput, g, 0, 1, True).to(dtype)
