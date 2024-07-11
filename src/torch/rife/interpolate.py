import importlib
import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F
from torch.fx.node import Argument, Target
from torch.library import custom_op, register_fake
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.conversion.converter_utils import (
    enforce_tensor_types,
    set_layer_name,
)
from torch_tensorrt.dynamo.types import TRTTensor


@custom_op("vsrife::upsample_nearest1d", mutates_args=())
def upsample_nearest1d(
    input: torch.Tensor,
    output_size: Optional[Sequence[int]],
    scale_factors: Optional[Sequence[float]],
) -> torch.Tensor:
    return torch._C._nn.upsample_nearest1d(input, output_size, scale_factors)


@register_fake("vsrife::upsample_nearest1d")
def upsample_nearest1d_fake(
    input: torch.Tensor,
    output_size: Optional[Sequence[int]],
    scale_factors: Optional[Sequence[float]],
) -> torch.Tensor:
    return torch._C._nn.upsample_nearest1d(input, output_size, scale_factors)


@custom_op("vsrife::upsample_nearest2d", mutates_args=())
def upsample_nearest2d(
    input: torch.Tensor,
    output_size: Optional[Sequence[int]],
    scale_factors: Optional[Sequence[float]],
) -> torch.Tensor:
    return torch._C._nn.upsample_nearest2d(input, output_size, scale_factors)


@register_fake("vsrife::upsample_nearest2d")
def upsample_nearest2d_fake(
    input: torch.Tensor,
    output_size: Optional[Sequence[int]],
    scale_factors: Optional[Sequence[float]],
) -> torch.Tensor:
    return torch._C._nn.upsample_nearest2d(input, output_size, scale_factors)


@custom_op("vsrife::upsample_nearest3d", mutates_args=())
def upsample_nearest3d(
    input: torch.Tensor,
    output_size: Optional[Sequence[int]],
    scale_factors: Optional[Sequence[float]],
) -> torch.Tensor:
    return torch._C._nn.upsample_nearest3d(input, output_size, scale_factors)


@register_fake("vsrife::upsample_nearest3d")
def upsample_nearest3d_fake(
    input: torch.Tensor,
    output_size: Optional[Sequence[int]],
    scale_factors: Optional[Sequence[float]],
) -> torch.Tensor:
    return torch._C._nn.upsample_nearest3d(input, output_size, scale_factors)


@custom_op("vsrife::upsample_linear1d", mutates_args=())
def upsample_linear1d(
    input: torch.Tensor,
    output_size: Optional[Sequence[int]],
    align_corners: bool,
    scale_factors: Optional[Sequence[float]],
) -> torch.Tensor:
    return torch._C._nn.upsample_linear1d(
        input, output_size, align_corners, scale_factors
    )


@register_fake("vsrife::upsample_linear1d")
def upsample_linear1d_fake(
    input: torch.Tensor,
    output_size: Optional[Sequence[int]],
    align_corners: bool,
    scale_factors: Optional[Sequence[float]],
) -> torch.Tensor:
    return torch._C._nn.upsample_linear1d(
        input, output_size, align_corners, scale_factors
    )


@custom_op("vsrife::upsample_bilinear2d", mutates_args=())
def upsample_bilinear2d(
    input: torch.Tensor,
    output_size: Optional[Sequence[int]],
    align_corners: bool,
    scale_factors: Optional[Sequence[float]],
) -> torch.Tensor:
    return torch._C._nn.upsample_bilinear2d(
        input, output_size, align_corners, scale_factors
    )


@register_fake("vsrife::upsample_bilinear2d")
def upsample_bilinear2d_fake(
    input: torch.Tensor,
    output_size: Optional[Sequence[int]],
    align_corners: bool,
    scale_factors: Optional[Sequence[float]],
) -> torch.Tensor:
    return torch._C._nn.upsample_bilinear2d(
        input, output_size, align_corners, scale_factors
    )


@custom_op("vsrife::upsample_trilinear3d", mutates_args=())
def upsample_trilinear3d(
    input: torch.Tensor,
    output_size: Optional[Sequence[int]],
    align_corners: bool,
    scale_factors: Optional[Sequence[float]],
) -> torch.Tensor:
    return torch._C._nn.upsample_trilinear3d(
        input, output_size, align_corners, scale_factors
    )


@register_fake("vsrife::upsample_trilinear3d")
def upsample_trilinear3d_fake(
    input: torch.Tensor,
    output_size: Optional[Sequence[int]],
    align_corners: bool,
    scale_factors: Optional[Sequence[float]],
) -> torch.Tensor:
    return torch._C._nn.upsample_trilinear3d(
        input, output_size, align_corners, scale_factors
    )


@custom_op("vsrife::upsample_bicubic2d", mutates_args=())
def upsample_bicubic2d(
    input: torch.Tensor,
    output_size: Optional[Sequence[int]],
    align_corners: bool,
    scale_factors: Optional[Sequence[float]],
) -> torch.Tensor:
    return torch._C._nn.upsample_bicubic2d(
        input, output_size, align_corners, scale_factors
    )


@register_fake("vsrife::upsample_bicubic2d")
def upsample_bicubic2d_fake(
    input: torch.Tensor,
    output_size: Optional[Sequence[int]],
    align_corners: bool,
    scale_factors: Optional[Sequence[float]],
) -> torch.Tensor:
    return torch._C._nn.upsample_bicubic2d(
        input, output_size, align_corners, scale_factors
    )


def upsample(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    size: Optional[Sequence[int]],
    scale_factor: Optional[Sequence[float]],
    mode: str,
    align_corners: bool,
) -> TRTTensor:
    layer = ctx.net.add_resize(input)

    if size is not None:
        layer.shape = list(input.shape)[:2] + list(size)
    else:
        layer.scales = [1.0, 1.0] + list(scale_factor)

    if mode == "nearest":
        layer.resize_mode = trt.InterpolationMode.NEAREST
        layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ASYMMETRIC
    elif mode in ("linear", "bilinear", "trilinear"):
        layer.resize_mode = trt.InterpolationMode.LINEAR
        layer.coordinate_transformation = (
            trt.ResizeCoordinateTransformation.ALIGN_CORNERS
            if align_corners
            else trt.ResizeCoordinateTransformation.HALF_PIXEL
        )
    elif mode == "bicubic":
        layer.resize_mode = trt.InterpolationMode.CUBIC
        layer.coordinate_transformation = (
            trt.ResizeCoordinateTransformation.ALIGN_CORNERS
            if align_corners
            else trt.ResizeCoordinateTransformation.HALF_PIXEL
        )

    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)


def args_bounds_check(
    args: Tuple[Argument, ...], i: int, replacement: Optional[Any] = None
) -> Any:
    return args[i] if len(args) > i and args[i] is not None else replacement


@dynamo_tensorrt_converter(torch.ops.vsrife.upsample_nearest1d.default)
@dynamo_tensorrt_converter(torch.ops.vsrife.upsample_nearest2d.default)
@dynamo_tensorrt_converter(torch.ops.vsrife.upsample_nearest3d.default)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def ops_upsample_nearest(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return upsample(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        size=args_bounds_check(args, 1),
        scale_factor=args_bounds_check(args, 2),
        mode="nearest",
        align_corners=False,
    )


@dynamo_tensorrt_converter(torch.ops.vsrife.upsample_linear1d.default)
@dynamo_tensorrt_converter(torch.ops.vsrife.upsample_bilinear2d.default)
@dynamo_tensorrt_converter(torch.ops.vsrife.upsample_trilinear3d.default)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def ops_upsample_linear(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return upsample(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        size=args_bounds_check(args, 1),
        scale_factor=args_bounds_check(args, 3),
        mode="linear",
        align_corners=args[2],
    )


@dynamo_tensorrt_converter(torch.ops.vsrife.upsample_bicubic2d.default)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
    }
)
def ops_upsample_bicubic(
    ctx: ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    return upsample(
        ctx,
        target,
        SourceIR.ATEN,
        name,
        args[0],
        size=args_bounds_check(args, 1),
        scale_factor=args_bounds_check(args, 3),
        mode="bicubic",
        align_corners=args[2],
    )


def _is_integer(x) -> bool:
    r"""Type check the input number is an integer.

    Will return True for int, SymInt, Numpy integers and Tensors with integer elements.
    """
    if isinstance(x, (int, torch.SymInt)):
        return True
    if isinstance(x, np.integer):
        return True
    return isinstance(x, torch.Tensor) and not x.is_floating_point()


def interpolate(
    input: torch.Tensor,
    size: Sequence[int] | None = None,
    scale_factor: float | None = None,
    mode: str = "nearest",
    align_corners: bool | None = None,
    recompute_scale_factor: bool | None = None,
    antialias: bool = False,
) -> torch.Tensor:
    if mode in ("nearest", "area", "nearest-exact"):
        if align_corners is not None:
            raise ValueError(
                "align_corners option can only be set with the "
                "interpolating modes: linear | bilinear | bicubic | trilinear"
            )
    else:
        if align_corners is None:
            align_corners = False

    dim = input.dim() - 2  # Number of spatial dimensions.

    # Process size and scale_factor.  Validate that exactly one is set.
    # Validate its length if it is a list, or expand it if it is a scalar.
    # After this block, exactly one of output_size and scale_factors will
    # be non-None, and it will be a list (or tuple).
    if size is not None and scale_factor is not None:
        raise ValueError("only one of size or scale_factor should be defined")
    elif size is not None:
        assert scale_factor is None
        scale_factors = None
        if len(size) != dim:
            raise ValueError(
                "Input and output must have the same number of spatial dimensions, but got "
                f"input with spatial dimensions of {list(input.shape[2:])} and output size of {size}. "
                "Please provide input tensor in (N, C, d1, d2, ...,dK) format and "
                "output size in (o1, o2, ...,oK) format."
            )
        if not torch.jit.is_scripting():
            if not all(_is_integer(x) for x in size):
                raise TypeError(
                    "expected size to be one of int or Tuple[int] or Tuple[int, int] or "
                    f"Tuple[int, int, int], but got size with types {[type(x) for x in size]}"
                )
        output_size = size
    elif scale_factor is not None:
        assert size is None
        output_size = None
        scale_factors = [scale_factor for _ in range(dim)]
    else:
        raise ValueError("either size or scale_factor should be defined")

    if (
        recompute_scale_factor is not None
        and recompute_scale_factor
        and size is not None
    ):
        raise ValueError(
            "recompute_scale_factor is not meaningful with an explicit size."
        )

    # "area" mode always requires an explicit size rather than scale factor.
    # Re-use the recompute_scale_factor code path.
    if mode == "area" and output_size is None:
        recompute_scale_factor = True

    if recompute_scale_factor is not None and recompute_scale_factor:
        # We compute output_size here, then un-set scale_factors.
        # The C++ code will recompute it based on the (integer) output size.
        assert scale_factors is not None
        if not torch.jit.is_scripting() and torch._C._get_tracing_state():
            # make scale_factor a tensor in tracing so constant doesn't get baked in
            output_size = [
                (
                    torch.floor(
                        (
                            input.size(i + 2).float()
                            * torch.tensor(scale_factors[i], dtype=torch.float32)
                        ).float()
                    )
                )
                for i in range(dim)
            ]
        elif torch.jit.is_scripting():
            output_size = [
                int(math.floor(float(input.size(i + 2)) * scale_factors[i]))
                for i in range(dim)
            ]
        else:
            output_size = [
                torch.sym_int(input.size(i + 2) * scale_factors[i]) for i in range(dim)
            ]
        scale_factors = None

    if antialias and not (mode in ("bilinear", "bicubic") and input.ndim == 4):
        raise ValueError(
            "Anti-alias option is restricted to bilinear and bicubic modes and requires a 4-D tensor as input"
        )

    if input.dim() == 3 and mode == "nearest":
        return torch.ops.vsrife.upsample_nearest1d(input, output_size, scale_factors)
    if input.dim() == 4 and mode == "nearest":
        return torch.ops.vsrife.upsample_nearest2d(input, output_size, scale_factors)
    if input.dim() == 5 and mode == "nearest":
        return torch.ops.vsrife.upsample_nearest3d(input, output_size, scale_factors)

    if input.dim() == 3 and mode == "nearest-exact":
        return torch._C._nn._upsample_nearest_exact1d(input, output_size, scale_factors)
    if input.dim() == 4 and mode == "nearest-exact":
        return torch._C._nn._upsample_nearest_exact2d(input, output_size, scale_factors)
    if input.dim() == 5 and mode == "nearest-exact":
        return torch._C._nn._upsample_nearest_exact3d(input, output_size, scale_factors)

    if input.dim() == 3 and mode == "area":
        assert output_size is not None
        return F.adaptive_avg_pool1d(input, output_size)
    if input.dim() == 4 and mode == "area":
        assert output_size is not None
        return F.adaptive_avg_pool2d(input, output_size)
    if input.dim() == 5 and mode == "area":
        assert output_size is not None
        return F.adaptive_avg_pool3d(input, output_size)

    if input.dim() == 3 and mode == "linear":
        assert align_corners is not None
        return torch.ops.vsrife.upsample_linear1d(
            input, output_size, align_corners, scale_factors
        )
    if input.dim() == 4 and mode == "bilinear":
        assert align_corners is not None
        if antialias:
            return torch._C._nn._upsample_bilinear2d_aa(
                input, output_size, align_corners, scale_factors
            )
        # Two levels are necessary to prevent TorchScript from touching
        # are_deterministic_algorithms_enabled.
        if not torch.jit.is_scripting():
            if torch.are_deterministic_algorithms_enabled() and input.is_cuda:
                # Use slow decomp whose backward will be in terms of index_put
                # importlib is required because the import cannot be top level
                # (cycle) and cannot be nested (TS doesn't support)
                return importlib.import_module(
                    "torch._decomp.decompositions"
                )._upsample_linear_vec(input, output_size, align_corners, scale_factors)
        return torch.ops.vsrife.upsample_bilinear2d(
            input, output_size, align_corners, scale_factors
        )
    if input.dim() == 5 and mode == "trilinear":
        assert align_corners is not None
        return torch.ops.vsrife.upsample_trilinear3d(
            input, output_size, align_corners, scale_factors
        )
    if input.dim() == 4 and mode == "bicubic":
        assert align_corners is not None
        if antialias:
            return torch._C._nn._upsample_bicubic2d_aa(
                input, output_size, align_corners, scale_factors
            )
        return torch.ops.vsrife.upsample_bicubic2d(
            input, output_size, align_corners, scale_factors
        )

    if input.dim() == 3 and mode == "bilinear":
        raise NotImplementedError("Got 3D input, but bilinear mode needs 4D input")
    if input.dim() == 3 and mode == "trilinear":
        raise NotImplementedError("Got 3D input, but trilinear mode needs 5D input")
    if input.dim() == 4 and mode == "linear":
        raise NotImplementedError("Got 4D input, but linear mode needs 3D input")
    if input.dim() == 4 and mode == "trilinear":
        raise NotImplementedError("Got 4D input, but trilinear mode needs 5D input")
    if input.dim() == 5 and mode == "linear":
        raise NotImplementedError("Got 5D input, but linear mode needs 3D input")
    if input.dim() == 5 and mode == "bilinear":
        raise NotImplementedError("Got 5D input, but bilinear mode needs 4D input")

    raise NotImplementedError(
        "Input Error: Only 3D, 4D and 5D input Tensors supported"
        f" (got {input.dim()}D) for the modes: nearest | linear | bilinear | bicubic | trilinear | area | nearest-exact"
        f" (got {mode})"
    )
