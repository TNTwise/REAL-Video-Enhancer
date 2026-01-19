from .utils.tools import *

if check_cupy_env():
    from ..models.softsplat.softsplat import softsplat as warp
else:
    print("System does not have CUDA installed, falling back to PyTorch")
    from ..models.softsplat.softsplat_torch import softsplat as warp


def get_drm_t(drm, t, precision=1e-3):
    """
    DRM is a tensor with dimensions b, 1, h, w, where for any value x (0 < x < 1).
    We define the timestep of the entire DRM tensor as 0.5, want the entire DRM approach t,
    but during this process, all values in the tensor must maintain their original proportions.

    Example:
        Input:
            drm = [0.1, 0.7, 0.4, 0.2, ...]
            init_t = 0.5
            target_t = 0.8

        Iteration Outputs:
            [0.1900(0.1 + (1 - 0.1) * 0.1), 0.9100, ...]  t = 0.5 + (1 - 0.5) * 0.5 = 0.75\n
            [0.2710(0.19 + (1 - 0.19) * 0.1), 0.9730, ...]  t = 0.75 + (1 - 0.75) * 0.5 = 0.875\n
            [0.2629(0.271 - (0.271 - 0.19) * 0.1), 0.9289, ...]  t = 0.875 - (0.875 - 0.75) * 0.5 = 0.8125\n
            [0.2556(0.2629 - (0.2629 - 0.19) * 0.1), 0.9157, ...]  t = 0.8125 - (0.8125 - 0.75) * 0.5 = 0.78125\n
            [0.2563(0.2556 + (0.2629 - 0.2556) * 0.1), 0.9249, ...]  t = 0.78125 + (0.8125 - 0.78125) * 0.5 = 0.796875\n
            [0.2570(0.2563 + (0.2629 - 0.2563) * 0.1), 0.9277, ...]  t = 0.796875 + (0.8125 - 0.796875) * 0.5 = 0.8046875\n
            ...

        Final Output:
            drm_t = [0.2569, 0.9258, 0.7106, 0.4486, ...]  t = 0.80078125
    """
    dtype = drm.dtype

    _x, b = 0.5, 0.5
    l, r = 0, 1

    # float is suggested for drm calculation to avoid overflow
    x_drm, b_drm = drm.float().clone(), drm.float().clone()
    l_drm, r_drm = x_drm.clone() * 0, x_drm.clone() * 0 + 1

    while abs(_x - t) > precision:
        if _x > t:
            r = _x
            # print(f"{_x} - ({_x} - {l}) * {b}")
            _x = _x - (_x - l) * b

            r_drm = x_drm.clone()
            x_drm = x_drm - (x_drm - l_drm) * b_drm
            # print(_x, x_drm)

        if _x < t:
            l = _x
            # print(f"{_x} + ({r} - {_x}) * {b}")
            _x = _x + (r - _x) * b

            l_drm = x_drm.clone()
            x_drm = x_drm + (r_drm - x_drm) * b_drm
            # print(_x, x_drm)

    return x_drm.to(dtype)


def calc_drm_rife(t, flow10, flow12, linear=False):
    # Compute the distance using the optical flow and distance calculator
    d10 = distance_calculator(flow10) + 1e-4
    d12 = distance_calculator(flow12) + 1e-4

    # Calculate the distance ratio map
    drm10 = d10 / (d10 + d12)
    drm12 = d12 / (d10 + d12)

    if linear:
        drm_t0_unaligned = drm10 * t * 2
        drm_t1_unaligned = drm12 * t * 2
    else:
        drm_t0_unaligned = get_drm_t(drm10, t)
        drm_t1_unaligned = get_drm_t(drm12, t)

    warp_method = "avg"
    # When using RIFE to generate intermediate frames between I0 and I1,
    # if the input image order is I0, I1, you need to use drm_t_I0_t01.
    # Conversely, if the order is reversed, you should use drm_t_I1_t01.
    # The same rule applies when processing intermediate frames between I1 and I2.

    # For RIFE, drm should be aligned with the time corresponding to the intermediate frame.
    # drm_t0_t01 = warp(drm_t0_unaligned, flow10 * drm_t0_unaligned, None, warp_method)
    drm_t1_t01 = warp(drm_t1_unaligned, flow10 * drm_t1_unaligned, None, warp_method)
    drm_t1_t12 = warp(drm_t0_unaligned, flow12 * drm_t0_unaligned, None, warp_method)
    # drm_t2_t12 = warp(drm_t1_unaligned, flow12 * drm_t1_unaligned, None, warp_method)

    ones_mask = drm10.clone() * 0 + 1

    mask_t1_t01 = warp(ones_mask, flow10 * drm_t1_unaligned, None, warp_method)
    mask_t1_t12 = warp(ones_mask, flow12 * drm_t0_unaligned, None, warp_method)

    gap_t1_t01 = mask_t1_t01 < 0.999
    gap_t1_t12 = mask_t1_t12 < 0.999

    drm_t1_t01[gap_t1_t01] = drm_t1_unaligned[gap_t1_t01]
    drm_t1_t12[gap_t1_t12] = drm_t0_unaligned[gap_t1_t12]

    return {"drm_t1_t01": drm_t1_t01, "drm_t1_t12": drm_t1_t12}


def calc_drm_gmfss(t, flow10, flow12, metric10, metric12, linear=False):
    # Compute the distance using the optical flow and distance calculator
    d10 = distance_calculator(flow10)
    d12 = distance_calculator(flow12)

    # Calculate the distance ratio map
    drm10 = d10 / (d10 + d12)
    drm12 = d12 / (d10 + d12)

    warp_method = "soft" if (metric10 is not None and metric12 is not None) else "avg"

    if linear:
        drm1t_t01 = drm12 * t * 2
        drm1t_t12 = drm10 * t * 2
        drm0t_t01_unaligned = 1 - drm1t_t01
        drm2t_t12_unaligned = 1 - drm1t_t12
    else:
        drm1t_t01 = get_drm_t(drm12, t)
        drm1t_t12 = get_drm_t(drm10, t)
        drm0t_t01_unaligned = 1 - drm1t_t01
        drm2t_t12_unaligned = 1 - drm1t_t12

    drm0t_t01 = warp(drm0t_t01_unaligned, flow10, metric10, warp_method)
    drm2t_t12 = warp(drm2t_t12_unaligned, flow12, metric12, warp_method)

    # Create a mask with all ones to identify the holes in the warped drm maps
    ones_mask = drm0t_t01.clone() * 0 + 1

    # Warp the ones mask
    warped_ones_mask0t_t01 = warp(ones_mask, flow10, metric10, warp_method)
    warped_ones_mask2t_t12 = warp(ones_mask, flow12, metric12, warp_method)

    # Identify holes in warped drm map
    gap_0t_t01 = warped_ones_mask0t_t01 < 0.999
    gap_2t_t12 = warped_ones_mask2t_t12 < 0.999

    # Fill the holes in the warped drm maps with the inverse of the original drm maps
    drm0t_t01[gap_0t_t01] = drm0t_t01_unaligned[gap_0t_t01]
    drm2t_t12[gap_2t_t12] = drm2t_t12_unaligned[gap_2t_t12]

    return {
        "drm0t_t01": drm0t_t01,
        "drm1t_t01": drm1t_t01,
        "drm1t_t12": drm1t_t12,
        "drm2t_t12": drm2t_t12,
    }


def calc_drm_rife_auxiliary(t, flow10, flow12, metric10, metric12, linear=False):
    # Compute the distance using the optical flow and distance calculator
    d10 = distance_calculator(flow10) + 1e-4
    d12 = distance_calculator(flow12) + 1e-4

    # Calculate the distance ratio map
    drm10 = d10 / (d10 + d12)
    drm12 = d12 / (d10 + d12)

    if linear:
        drm_t0_unaligned = drm10 * t * 2
        drm_t1_unaligned = drm12 * t * 2
    else:
        drm_t0_unaligned = get_drm_t(drm10, t)
        drm_t1_unaligned = get_drm_t(drm12, t)

    warp_method = "soft" if (metric10 is not None and metric12 is not None) else "avg"

    # For RIFE, drm should be aligned with the time corresponding to the intermediate frame.
    drm_t1_t01 = warp(
        drm_t1_unaligned, flow10 * drm_t1_unaligned, metric10, warp_method
    )
    drm_t1_t12 = warp(
        drm_t0_unaligned, flow12 * drm_t0_unaligned, metric12, warp_method
    )

    ones_mask = drm10.clone() * 0 + 1

    mask_t1_t01 = warp(ones_mask, flow10 * drm_t1_unaligned, metric10, warp_method)
    mask_t1_t12 = warp(ones_mask, flow12 * drm_t0_unaligned, metric12, warp_method)

    gap_t1_t01 = mask_t1_t01 < 0.999
    gap_t1_t12 = mask_t1_t12 < 0.999

    drm_t1_t01[gap_t1_t01] = drm_t1_unaligned[gap_t1_t01]
    drm_t1_t12[gap_t1_t12] = drm_t0_unaligned[gap_t1_t12]

    # why use drm0t1 not drm1t0, because rife use backward warp not forward warp.
    return {"drm_t1_t01": drm_t1_t01, "drm_t1_t12": drm_t1_t12}
