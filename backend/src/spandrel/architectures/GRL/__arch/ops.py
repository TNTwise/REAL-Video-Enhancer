from __future__ import annotations

from math import prod

import numpy as np
import torch

from spandrel.util.timm import to_2tuple


def bchw_to_bhwc(x: torch.Tensor) -> torch.Tensor:
    """Permutes a tensor from the shape (B, C, H, W) to (B, H, W, C)."""
    return x.permute(0, 2, 3, 1)


def bchw_to_blc(x: torch.Tensor) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, C, H, W) to (B, L, C)."""
    return x.flatten(2).transpose(1, 2)


def blc_to_bchw(x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, L, C) to (B, C, H, W)."""
    B, _L, C = x.shape
    return x.transpose(1, 2).view(B, C, *x_size)


def blc_to_bhwc(x: torch.Tensor, x_size: tuple[int, int] | list[int]) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, L, C) to (B, H, W, C)."""
    B, _L, C = x.shape
    return x.view(B, *x_size, C)


def window_partition(x, window_size: tuple[int, int] | list[int]):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(
        B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size[0], window_size[1], C)
    )
    return windows


def window_reverse(
    windows, window_size: tuple[int, int] | list[int], img_size: tuple[int, int]
):
    """
    Args:
        windows: (num_windows * B, window_size[0], window_size[1], C)
        window_size (Tuple[int, int]): Window size
        img_size (Tuple[int, int]): Image size

    Returns:
        x: (B, H, W, C)
    """
    H, W = img_size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(
        B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def _fill_window(
    input_resolution, window_size: tuple[int, int] | list[int], shift_size=None
):
    if shift_size is None:
        shift_size = [s // 2 for s in window_size]

    img_mask = torch.zeros((1, *input_resolution, 1))  # 1 H W 1
    h_slices = (
        slice(0, -window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    )
    w_slices = (
        slice(0, -window_size[1]),
        slice(-window_size[1], -shift_size[1]),
        slice(-shift_size[1], None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    # nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, prod(window_size))
    return mask_windows


#####################################
# Different versions of the functions
# 1) Swin Transformer, SwinIR, Square window attention in GRL;
# 2) Early development of the decomposition-based efficient attention mechanism (efficient_win_attn.py);
# 3) GRL. Window-anchor attention mechanism.
# 1) & 3) are still useful
#####################################


def calculate_mask(input_resolution, window_size, shift_size):
    """
    Use case: 1)
    """
    # calculate attention mask for SW-MSA
    if isinstance(shift_size, int):
        shift_size = to_2tuple(shift_size)
    mask_windows = _fill_window(input_resolution, window_size, shift_size)

    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
        attn_mask == 0, 0.0
    )  # nW, window_size**2, window_size**2

    return attn_mask


def calculate_mask_all(
    input_resolution,
    window_size,
    shift_size,
    anchor_window_down_factor=1,
    window_to_anchor=True,
):
    """
    Use case: 3)
    """
    # calculate attention mask for SW-MSA
    anchor_resolution = [s // anchor_window_down_factor for s in input_resolution]
    aws = [s // anchor_window_down_factor for s in window_size]
    anchor_shift = [s // anchor_window_down_factor for s in shift_size]

    # mask of window1: nW, Wh**Ww
    mask_windows = _fill_window(input_resolution, window_size, shift_size)
    # mask of window2: nW, AWh*AWw
    mask_anchor = _fill_window(anchor_resolution, aws, anchor_shift)

    if window_to_anchor:
        attn_mask = mask_windows.unsqueeze(2) - mask_anchor.unsqueeze(1)
    else:
        attn_mask = mask_anchor.unsqueeze(2) - mask_windows.unsqueeze(1)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
        attn_mask == 0, 0.0
    )  # nW, Wh**Ww, AWh*AWw

    return attn_mask


def _get_meshgrid_coords(
    start_coords: tuple[int, int] | list[int],
    end_coords: tuple[int, int] | list[int],
):
    coord_h = torch.arange(start_coords[0], end_coords[0])
    coord_w = torch.arange(start_coords[1], end_coords[1])
    coords = torch.stack(torch.meshgrid([coord_h, coord_w], indexing="ij"))  # 2, Wh, Ww
    coords = torch.flatten(coords, 1)  # 2, Wh*Ww
    return coords


def get_relative_coords_table_all(
    window_size: tuple[int, int] | list[int],
    pretrained_window_size: tuple[int, int] | list[int] = [0, 0],
    anchor_window_down_factor=1,
):
    """
    Use case: 3)

    Support all window shapes.
    Args:
        window_size:
        pretrained_window_size:
        anchor_window_down_factor:

    Returns:

    """
    # get relative_coords_table
    ws = window_size
    aws = [w // anchor_window_down_factor for w in window_size]
    pws = pretrained_window_size
    paws = [w // anchor_window_down_factor for w in pretrained_window_size]

    # positive table size: (Ww - 1) - (Ww - AWw) // 2
    ts_p = [w1 - 1 - (w1 - w2) // 2 for w1, w2 in zip(ws, aws)]
    # negative table size: -(AWw - 1) - (Ww - AWw) // 2
    ts_n = [-(w2 - 1) - (w1 - w2) // 2 for w1, w2 in zip(ws, aws)]
    pts = [w1 - 1 - (w1 - w2) // 2 for w1, w2 in zip(pws, paws)]

    # TODO: pretrained window size and pretrained anchor window size is only used here.
    # TODO: Investigate whether it is really important to use this setting when finetuning large window size
    # TODO: based on pretrained weights with small window size.

    coord_h = torch.arange(ts_n[0], ts_p[0] + 1, dtype=torch.float32)
    coord_w = torch.arange(ts_n[1], ts_p[1] + 1, dtype=torch.float32)
    table = torch.stack(torch.meshgrid([coord_h, coord_w], indexing="ij")).permute(
        1, 2, 0
    )
    table = table.contiguous().unsqueeze(0)  # 1, Wh+AWh-1, Ww+AWw-1, 2
    if pts[0] > 0:
        table[:, :, :, 0] /= pts[0]
        table[:, :, :, 1] /= pts[1]
    else:
        table[:, :, :, 0] /= ts_p[0]
        table[:, :, :, 1] /= ts_p[1]
    table *= 8  # normalize to -8, 8
    table = torch.sign(table) * torch.log2(torch.abs(table) + 1.0) / np.log2(8)
    # 1, Wh+AWh-1, Ww+AWw-1, 2
    return table


def coords_diff_odd(coords1, coords2, start_coord, max_diff):
    # The coordinates starts from (-start_coord[0], -start_coord[1])
    coords = coords1[:, :, None] - coords2[:, None, :]  # 2, Wh*Ww, AWh*AWw
    coords = coords.permute(1, 2, 0).contiguous()  # Wh*Ww, AWh*AWw, 2
    coords[:, :, 0] += start_coord[0]  # shift to start from 0
    coords[:, :, 1] += start_coord[1]
    coords[:, :, 0] *= max_diff
    idx = coords.sum(-1)  # Wh*Ww, AWh*AWw
    return idx


def get_relative_position_index_simple(
    window_size: tuple[int, int] | list[int],
    anchor_window_down_factor=1,
    window_to_anchor=True,
):
    """
    Use case: 3)
    This is a simplified version of get_relative_position_index_all
    The start coordinate of anchor window is also (0, 0)
    get pair-wise relative position index for each token inside the window
    """
    ws = window_size
    aws = [w // anchor_window_down_factor for w in window_size]

    coords = _get_meshgrid_coords((0, 0), window_size)  # 2, Wh*Ww
    coords_anchor = _get_meshgrid_coords((0, 0), aws)
    # 2, AWh*AWw

    max_horizontal_diff = aws[1] + ws[1] - 1
    if window_to_anchor:
        offset = [w2 - 1 for w2 in aws]
        idx = coords_diff_odd(coords, coords_anchor, offset, max_horizontal_diff)
    else:
        offset = [w1 - 1 for w1 in ws]
        idx = coords_diff_odd(coords_anchor, coords, offset, max_horizontal_diff)
    return idx  # Wh*Ww, AWh*AWw or AWh*AWw, Wh*Ww
