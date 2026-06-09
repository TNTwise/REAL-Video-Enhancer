# https://github.com/TNTwise/REAL-Video-Enhancer/blob/2.0/backend/src/pytorch/InterpolateArchs/util/softsplat_torch.py
# Thanks to TNTBad and ChatGPT
# Thanks to Nilas and ChatGPT

import torch

##########################################################
device = "cuda" if torch.cuda.is_available() else "cpu"

grid_cache = {}
out_cache = {}
linear_cache = {}
torch.set_float32_matmul_precision("medium")
torch.set_grad_enabled(False)

##########################################################


@torch.inference_mode()
def softsplat(
    tenIn: torch.Tensor,
    tenFlow: torch.Tensor,
    tenMetric: torch.Tensor,
    strMode: str,
):
    mode_parts = strMode.split("-")
    mode_main = mode_parts[0]
    mode_sub = mode_parts[1] if len(mode_parts) > 1 else None

    assert mode_main in ["sum", "avg", "linear", "soft"]
    if mode_main in ["sum", "avg"]:
        assert tenMetric is None
    if mode_main in ["linear", "soft"]:
        assert tenMetric is not None

    # Sanity: for 'linear'/'soft' we require matching spatial shapes
    if mode_main in ["linear", "soft"]:
        if tenIn.shape[2:] != tenMetric.shape[2:]:
            raise ValueError(
                f"softsplat: mismatched spatial sizes between input {tenIn.shape} and metric {tenMetric.shape}"
            )

    # Precompute exp once (bitwise identical vs computing twice)
    metric_exp = tenMetric.exp() if mode_main == "soft" else None

    mode_to_operation = {
        "avg": lambda: torch.cat(
            [
                tenIn,
                tenIn.new_ones(
                    [
                        tenIn.shape[0],
                        1,
                        tenIn.shape[2],
                        tenIn.shape[3],
                    ]
                ),
            ],
            1,
        ),
        "linear": lambda: torch.cat([tenIn * tenMetric, tenMetric], 1),
        "soft": lambda: torch.cat([tenIn * metric_exp, metric_exp], 1),
    }

    if mode_main in mode_to_operation:
        tenIn = mode_to_operation[mode_main]()

    tenOut = softsplat_func.apply(tenIn, tenFlow)

    if mode_main in ["avg", "linear", "soft"]:
        tenNormalize = tenOut[:, -1:, :, :]

        normalize_modes = {
            None: lambda x: x + 0.0000001,
            "addeps": lambda x: x + 0.0000001,
            "zeroeps": lambda x: torch.where(
                x == 0.0, torch.tensor(1.0, device=x.device), x
            ),
            "clipeps": lambda x: x.clip(0.0000001, None),
        }

        if mode_sub in normalize_modes:
            tenNormalize = normalize_modes[mode_sub](tenNormalize)

        tenOut = tenOut[:, :-1, :, :] / tenNormalize

    return tenOut


class softsplat_func(torch.autograd.Function):
    """Custom autograd function implementing soft forward splatting.

    This version keeps the exact mathematical behaviour of the original
    implementation while:
      * Removing the Python loop over the 4 bilinear neighbour contributions.
      * Using a single vectorised index_add_ call to accumulate all weights.
      * Fixing a subtle caching bug where grids were reused across different batch sizes (N).
      * Reducing tensor permutations / reshapes and unnecessary temporaries.
    """

    @staticmethod
    @torch.inference_mode()
    @torch.amp.custom_fwd(device_type=device)
    def forward(ctx, tenIn: torch.Tensor, tenFlow: torch.Tensor):
        # Shapes / device
        N, C, H, W = tenIn.shape
        dev = tenIn.device
        origdtype = tenIn.dtype

        # Grid cache (independent of N to allow varying batch sizes)
        key = (H, W, dev, origdtype)
        if key not in grid_cache:
            gy, gx = torch.meshgrid(
                torch.arange(H, device=dev, dtype=origdtype),
                torch.arange(W, device=dev, dtype=origdtype),
                indexing="ij",
            )
            grid_cache[key] = (gy[None, None], gx[None, None])  # (1,1,H,W)
        gy, gx = grid_cache[key]
        gy = gy.expand(N, 1, H, W)
        gx = gx.expand(N, 1, H, W)

        # Target coords
        fltX = gx + tenFlow[:, 0:1]
        fltY = gy + tenFlow[:, 1:2]

        # Flatten
        fltX_flat = fltX.reshape(-1)
        fltY_flat = fltY.reshape(-1)
        feats_flat = tenIn.permute(0, 2, 3, 1).reshape(-1, C)

        # Finite mask
        mask = torch.isfinite(fltX_flat) & torch.isfinite(fltY_flat)
        if not mask.any():
            return torch.zeros_like(tenIn)
        fltX_flat = fltX_flat[mask]
        fltY_flat = fltY_flat[mask]
        feats_flat = feats_flat[mask]

        # Batch index for each surviving pixel
        linear_key = (N, H, W, dev)
        linear_full = linear_cache.get(linear_key)
        if linear_full is None or linear_full.numel() != N * H * W:
            linear_full = torch.arange(N * H * W, device=dev, dtype=torch.int64)
            linear_cache[linear_key] = linear_full
        batch_idx = (linear_full[mask]) // (H * W)

        # Corner integer coords
        x0 = torch.floor(fltX_flat)
        y0 = torch.floor(fltY_flat)
        x1 = x0 + 1
        y1 = y0 + 1
        x0l = x0.long()
        y0l = y0.long()
        x1l = x1.long()
        y1l = y1.long()

        w00 = (x1 - fltX_flat) * (y1 - fltY_flat)
        w10 = (fltX_flat - x0) * (y1 - fltY_flat)
        w01 = (x1 - fltX_flat) * (fltY_flat - y0)
        w11 = (fltX_flat - x0) * (fltY_flat - y0)

        m00 = (x0l >= 0) & (x0l < W) & (y0l >= 0) & (y0l < H)
        m10 = (x1l >= 0) & (x1l < W) & (y0l >= 0) & (y0l < H)
        m01 = (x0l >= 0) & (x0l < W) & (y1l >= 0) & (y1l < H)
        m11 = (x1l >= 0) & (x1l < W) & (y1l >= 0) & (y1l < H)

        base = batch_idx * (H * W)
        idx00 = base + y0l * W + x0l
        idx10 = base + y0l * W + x1l
        idx01 = base + y1l * W + x0l
        idx11 = base + y1l * W + x1l

        indices = torch.cat([idx00[m00], idx10[m10], idx01[m01], idx11[m11]], 0)
        values = torch.cat(
            [
                feats_flat[m00] * w00[m00].unsqueeze(1),
                feats_flat[m10] * w10[m10].unsqueeze(1),
                feats_flat[m01] * w01[m01].unsqueeze(1),
                feats_flat[m11] * w11[m11].unsqueeze(1),
            ],
            0,
        )

        cache_key = (N, C, H, W, dev, origdtype)
        out = out_cache.get(cache_key)
        if out is None or out.numel() != N * H * W * C:
            out = torch.zeros(N * H * W, C, device=dev, dtype=origdtype)
            out_cache[cache_key] = out
        else:
            out.zero_()
        out.index_add_(0, indices, values)
        return out.view(N, H, W, C).permute(0, 3, 1, 2)

    # Note: backward not implemented (function is inference-only by design).
