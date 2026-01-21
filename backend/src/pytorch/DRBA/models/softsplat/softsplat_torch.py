# torch fallback for softsplat inference
# https://github.com/98mxr/GMFSS_Fortuna/pull/11/files
# author: TNTwise
# https://github.com/TNTwise

import torch

##########################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'

grid_cache = {}
batch_cache = {}
torch.set_float32_matmul_precision('medium')
torch.set_grad_enabled(False)


##########################################################


@torch.inference_mode()
def softsplat(
    tenIn: torch.Tensor,
    tenFlow: torch.Tensor,
    tenMetric: torch.Tensor,
    strMode: str,
):
    mode_parts = strMode.split('-')
    mode_main = mode_parts[0]
    mode_sub = mode_parts[1] if len(mode_parts) > 1 else None

    assert mode_main in ['sum', 'avg', 'linear', 'soft']
    if mode_main in ['sum', 'avg']:
        assert tenMetric is None
    if mode_main in ['linear', 'soft']:
        assert tenMetric is not None

    mode_to_operation = {
        'avg': lambda: torch.cat(
            [
                tenIn,
                tenIn.new_ones([
                    tenIn.shape[0],
                    1,
                    tenIn.shape[2],
                    tenIn.shape[3],
                ]),
            ],
            1,
        ),
        'linear': lambda: torch.cat([tenIn * tenMetric, tenMetric], 1),
        'soft': lambda: torch.cat(
            [tenIn * tenMetric.exp(), tenMetric.exp()], 1
        ),
    }

    if mode_main in mode_to_operation:
        tenIn = mode_to_operation[mode_main]()

    tenOut = softsplat_func.apply(tenIn, tenFlow)

    if mode_main in ['avg', 'linear', 'soft']:
        tenNormalize = tenOut[:, -1:, :, :]

        normalize_modes = {
            None: lambda x: x + 0.0000001,
            'addeps': lambda x: x + 0.0000001,
            'zeroeps': lambda x: torch.where(
                x == 0.0, torch.tensor(1.0, device=x.device), x
            ),
            'clipeps': lambda x: x.clip(0.0000001, None),
        }

        if mode_sub in normalize_modes:
            tenNormalize = normalize_modes[mode_sub](tenNormalize)

        tenOut = tenOut[:, :-1, :, :] / tenNormalize

    return tenOut


class softsplat_func(torch.autograd.Function):
    @staticmethod
    @torch.inference_mode()
    @torch.amp.custom_fwd(device_type=device)
    def forward(ctx, tenIn, tenFlow):
        """
        Forward pass of the Softsplat function.

        Parameters:
            tenIn (torch.Tensor): Input tensor of shape [N, C, H, W]
            tenFlow (torch.Tensor): Flow tensor of shape [N, 2, H, W]

        Returns:
            torch.Tensor: Output tensor of shape [N, C, H, W]
        """
        N, C, H, W = tenIn.size()
        device = tenIn.device
        origdtype = tenIn.dtype

        # Initialize output tensor
        tenOut = torch.zeros_like(tenIn)

        key = (H, W, device, origdtype)
        if key not in grid_cache:
            # Create meshgrid of pixel coordinates
            gridY, gridX = torch.meshgrid(
                torch.arange(H, device=device, dtype=origdtype),
                torch.arange(W, device=device, dtype=origdtype),
                indexing='ij',
            )  # [H, W]
            # Cache the grids
            grid_cache[key] = (
                gridY.unsqueeze(0).unsqueeze(0).expand(N, 1, H, W),
                gridX.unsqueeze(0).unsqueeze(0).expand(N, 1, H, W),
            )

        if key not in batch_cache:
            batch_cache[key] = (
                torch
                .arange(N, device=device)
                .view(N, 1, 1)
                .expand(N, H, W)
                .reshape(-1)
            )

        gridY, gridX = grid_cache[key]
        batch_indices = batch_cache[key]

        # Compute fltX and fltY
        fltX = gridX + tenFlow[:, 0:1, :, :]
        fltY = gridY + tenFlow[:, 1:2, :, :]

        # Flatten variables
        fltX_flat = fltX.reshape(-1)
        fltY_flat = fltY.reshape(-1)
        tenIn_flat = tenIn.permute(0, 2, 3, 1).reshape(-1, C)

        # Finite mask
        finite_mask = torch.isfinite(fltX_flat) & torch.isfinite(fltY_flat)
        if not finite_mask.any():
            return tenOut

        fltX_flat = fltX_flat[finite_mask]
        fltY_flat = fltY_flat[finite_mask]
        tenIn_flat = tenIn_flat[finite_mask]
        batch_indices = batch_indices[finite_mask]

        # Compute integer positions
        intNW_X = torch.floor(fltX_flat).to(dtype=torch.int32)
        intNW_Y = torch.floor(fltY_flat).to(dtype=torch.int32)
        intNE_X = intNW_X + 1
        intNE_Y = intNW_Y
        intSW_X = intNW_X
        intSW_Y = intNW_Y + 1
        intSE_X = intNW_X + 1
        intSE_Y = intNW_Y + 1

        # Compute weights
        fltNW = (intSE_X - fltX_flat) * (intSE_Y - fltY_flat)
        fltNE = (fltX_flat - intSW_X) * (intSW_Y - fltY_flat)
        fltSW = (intNE_X - fltX_flat) * (fltY_flat - intNE_Y)
        fltSE = (fltX_flat - intNW_X) * (fltY_flat - intNW_Y)

        # Prepare output tensor flat
        tenOut_flat = tenOut.permute(0, 2, 3, 1).reshape(-1, C)

        # Define positions and weights
        positions = [
            (intNW_X, intNW_Y, fltNW),
            (intNE_X, intNE_Y, fltNE),
            (intSW_X, intSW_Y, fltSW),
            (intSE_X, intSE_Y, fltSE),
        ]

        H, W = int(H), int(W)

        for intX, intY, weight in positions:
            # Valid indices within image bounds
            valid_mask = (intX >= 0) & (intX < W) & (intY >= 0) & (intY < H)
            if not valid_mask.any():
                continue

            idx_b = batch_indices[valid_mask]
            idx_x = intX[valid_mask]
            idx_y = intY[valid_mask]
            w = weight[valid_mask]
            vals = tenIn_flat[valid_mask] * w.unsqueeze(1)

            # Compute linear indices
            idx_NHW = idx_b * H * W + idx_y * W + idx_x

            # Accumulate values using index_add_
            tenOut_flat.index_add_(0, idx_NHW, vals)

        # Reshape tenOut back to [N, C, H, W]
        tenOut = tenOut_flat.view(N, H, W, C).permute(0, 3, 1, 2)

        return tenOut

    # end

    # end

    # end
