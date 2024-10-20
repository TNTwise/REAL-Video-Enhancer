"""
MIT License

Copyright (c) 2021 HolyWu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from typing import Self, Sequence, Union

import cupy as cp
import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F
from torch.fx.node import Argument, Target
from torch.library import custom_op, register_fake
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.conversion.converter_utils import (
    enforce_tensor_types,
    set_layer_name,
)
from torch_tensorrt.dynamo.types import TRTTensor


class WarpPlugin(
    trt.IPluginV3, trt.IPluginV3OneCore, trt.IPluginV3OneBuild, trt.IPluginV3OneRuntime
):
    def __init__(self) -> None:
        trt.IPluginV3.__init__(self)
        trt.IPluginV3OneCore.__init__(self)
        trt.IPluginV3OneBuild.__init__(self)
        trt.IPluginV3OneRuntime.__init__(self)
        self.num_outputs = 1
        self.plugin_name = "WarpPlugin"
        self.plugin_version = "1"
        self.plugin_namespace = ""

    def clone(self) -> WarpPlugin:
        cloned = WarpPlugin()
        cloned.__dict__.update(self.__dict__)
        return cloned

    def get_capability_interface(self, type: trt.PluginCapabilityType) -> Self:
        return self

    def configure_plugin(
        self,
        inp: list[trt.DynamicPluginTensorDesc],
        out: list[trt.DynamicPluginTensorDesc],
    ) -> None:
        pass

    def get_output_data_types(
        self, input_types: list[trt.DataType]
    ) -> list[trt.DataType]:
        return [input_types[0]]

    def get_output_shapes(
        self,
        inputs: list[trt.DimsExprs],
        shape_inputs: list[trt.DimsExprs],
        expr_builder: trt.IExprBuilder,
    ) -> list[trt.DimsExprs]:
        return [inputs[0]]

    def supports_format_combination(
        self, pos: int, in_out: list[trt.DynamicPluginTensorDesc], num_inputs: int
    ) -> bool:
        assert pos < len(in_out)
        assert num_inputs == 4

        desc = in_out[pos].desc
        return (
            desc.format == trt.TensorFormat.LINEAR and desc.type == trt.DataType.FLOAT
        )

    def attach_to_context(
        self, resource_context: trt.IPluginResourceContext
    ) -> WarpPlugin:
        return self.clone()

    def enqueue(
        self,
        input_desc: list[trt.PluginTensorDesc],
        output_desc: list[trt.PluginTensorDesc],
        inputs: list[int],
        outputs: list[int],
        workspace: int,
        stream: int,
    ) -> None:
        dtype = trt.nptype(input_desc[0].type)
        itemsize = cp.dtype(dtype).itemsize

        with cp.cuda.ExternalStream(stream):
            input0_mem = cp.cuda.UnownedMemory(
                inputs[0], np.prod(input_desc[0].dims) * itemsize, self
            )
            input1_mem = cp.cuda.UnownedMemory(
                inputs[1], np.prod(input_desc[1].dims) * itemsize, self
            )
            input2_mem = cp.cuda.UnownedMemory(
                inputs[2], np.prod(input_desc[2].dims) * itemsize, self
            )
            input3_mem = cp.cuda.UnownedMemory(
                inputs[3], np.prod(input_desc[3].dims) * itemsize, self
            )
            output_mem = cp.cuda.UnownedMemory(
                outputs[0], np.prod(output_desc[0].dims) * itemsize, self
            )

            input0_ptr = cp.cuda.MemoryPointer(input0_mem, 0)
            input1_ptr = cp.cuda.MemoryPointer(input1_mem, 0)
            input2_ptr = cp.cuda.MemoryPointer(input2_mem, 0)
            input3_ptr = cp.cuda.MemoryPointer(input3_mem, 0)
            output_ptr = cp.cuda.MemoryPointer(output_mem, 0)

            input0_d = cp.ndarray(input_desc[0].dims, dtype=dtype, memptr=input0_ptr)
            input1_d = cp.ndarray(input_desc[1].dims, dtype=dtype, memptr=input1_ptr)
            input2_d = cp.ndarray(input_desc[2].dims, dtype=dtype, memptr=input2_ptr)
            input3_d = cp.ndarray(input_desc[3].dims, dtype=dtype, memptr=input3_ptr)
            output_d = cp.ndarray(
                (np.prod(output_desc[0].dims),), dtype=dtype, memptr=output_ptr
            )

            input0_t = torch.as_tensor(input0_d)
            input1_t = torch.as_tensor(input1_d)
            input2_t = torch.as_tensor(input2_d)
            input3_t = torch.as_tensor(input3_d)

            out = warp_custom(input0_t, input1_t, input2_t, input3_t)
            cp.copyto(output_d, cp.reshape(cp.asarray(out), (-1,)))

    def get_fields_to_serialize(self) -> trt.PluginFieldCollection_:
        return trt.PluginFieldCollection()

    def on_shape_change(
        self, inp: list[trt.PluginTensorDesc], out: list[trt.PluginTensorDesc]
    ) -> None:
        pass

    def set_tactic(self, tactic: int) -> None:
        pass


class WarpPluginCreator(trt.IPluginCreatorV3One):
    def __init__(self) -> None:
        super().__init__()
        self.name = "WarpPlugin"
        self.plugin_version = "1"
        self.plugin_namespace = ""
        self.field_names = trt.PluginFieldCollection()

    def create_plugin(
        self,
        name: str,
        field_collection: trt.PluginFieldCollection_,
        phase: trt.TensorRTPhase,
    ) -> WarpPlugin:
        return WarpPlugin()


@custom_op("vsrife::warp", mutates_args=())
def warp_custom(
    tenInput: torch.Tensor,
    tenFlow: torch.Tensor,
    tenFlow_div: torch.Tensor,
    backwarp_tenGrid: torch.Tensor,
) -> torch.Tensor:
    tenFlow = torch.cat(
        [tenFlow[:, 0:1] / tenFlow_div[0], tenFlow[:, 1:2] / tenFlow_div[1]], 1
    )
    g = (backwarp_tenGrid + tenFlow).permute(0, 2, 3, 1)
    return F.grid_sample(
        input=tenInput,
        grid=g,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


@register_fake("vsrife::warp")
def warp_fake(
    tenInput: torch.Tensor,
    tenFlow: torch.Tensor,
    tenFlow_div: torch.Tensor,
    backwarp_tenGrid: torch.Tensor,
) -> torch.Tensor:
    return tenInput


@dynamo_tensorrt_converter(torch.ops.vsrife.warp.default, supports_dynamic_shapes=True)
@enforce_tensor_types(
    {
        0: (TRTTensor,),
        1: (TRTTensor,),
        2: (TRTTensor,),
        3: (TRTTensor,),
    }
)
def ops_warp(
    ctx: ConversionContext,
    target: Target,
    args: tuple[Argument, ...],
    kwargs: dict[str, Argument],
    name: str,
) -> Union[TRTTensor, Sequence[TRTTensor]]:
    creator = trt.get_plugin_registry().get_creator("WarpPlugin", version="1")
    field_collection = trt.PluginFieldCollection()
    plugin = creator.create_plugin(
        "WarpPlugin", field_collection=field_collection, phase=trt.TensorRTPhase.BUILD
    )
    layer = ctx.net.add_plugin_v3(inputs=list(args), shape_inputs=[], plugin=plugin)
    set_layer_name(layer, target, name)
    return layer.get_output(0)


def warp(
    tenInput: torch.Tensor,
    tenFlow: torch.Tensor,
    tenFlow_div: torch.Tensor,
    backwarp_tenGrid: torch.Tensor,
) -> torch.Tensor:
    dtype = tenInput.dtype
    tenInput = tenInput.to(torch.float)
    tenFlow = tenFlow.to(torch.float)

    return torch.ops.vsrife.warp(tenInput, tenFlow, tenFlow_div, backwarp_tenGrid).to(
        dtype
    )
