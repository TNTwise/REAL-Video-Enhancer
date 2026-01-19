import torch
import torch.nn.functional as F

coords_grid_cache = {}


def coords_grid(
    b, h, w, homogeneous=False, device=None, dtype: torch.dtype = torch.float32
):
    k = (str(device), str((b, h, w)))
    if k in coords_grid_cache:
        return coords_grid_cache[k]
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0)  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device, dtype=dtype)
    coords_grid_cache[k] = grid
    return grid


window_grid_cache = {}


def generate_window_grid(
    h_min, h_max, w_min, w_max, len_h, len_w, device=None, dtype=torch.float32
):
    assert device is not None
    k = (str(device), str((h_min, h_max, w_min, w_max, len_h, len_w)))
    if k in window_grid_cache:
        return window_grid_cache[k]
    x, y = torch.meshgrid(
        [
            torch.linspace(w_min, w_max, len_w, device=device),
            torch.linspace(h_min, h_max, len_h, device=device),
        ],
    )
    grid = torch.stack((x, y), -1).transpose(0, 1).to(device, dtype=dtype)  # [H, W, 2]
    window_grid_cache[k] = grid
    return grid


normalize_coords_cache = {}


def normalize_coords(coords, h, w):
    # coords: [B, H, W, 2]
    k = (str(coords.device), str((h, w)))
    if k in normalize_coords_cache:
        c = normalize_coords_cache[k]
    else:
        c = torch.tensor(
            [(w - 1) / 2.0, (h - 1) / 2.0], dtype=coords.dtype, device=coords.device
        )
        normalize_coords_cache[k] = c
    return (coords - c) / c  # [-1, 1]


def bilinear_sample(
    img, sample_coords, mode="bilinear", padding_mode="zeros", return_mask=False
):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(
        img, grid, mode=mode, padding_mode=padding_mode, align_corners=True
    )

    if return_mask:
        mask = (
            (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)
        )  # [B, H, W]

        return img, mask

    return img


def flow_warp(feature, flow, mask=False, padding_mode="zeros"):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = (
        coords_grid(b, h, w, device=flow.device, dtype=flow.dtype) + flow
    )  # [B, 2, H, W]

    return bilinear_sample(feature, grid, padding_mode=padding_mode, return_mask=mask)


def forward_backward_consistency_check(fwd_flow, bwd_flow, alpha=0.01, beta=0.5):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    # alpha and beta values are following UnFlow (https://arxiv.org/abs/1711.07837)
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)  # [B, H, W]

    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow)  # [B, 2, H, W]
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow)  # [B, 2, H, W]

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)  # [B, H, W]
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd > threshold).to(fwd_flow)  # [B, H, W]
    bwd_occ = (diff_bwd > threshold).to(bwd_flow)

    return fwd_occ, bwd_occ
