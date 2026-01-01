"""
MIT License

Copyright (c) 2024 TNTwise

cPermission is hereby granted, free of charge, to any person obtaining a copy
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

import sys
import os
from ..utils.Util import suppress_stdout_stderr, warnAndLog, log
from ..version import __version__
import time
with suppress_stdout_stderr():
    import torch
    import torch_tensorrt
    import tensorrt as trt
    from torch._export.converter import TS2EPConverter
    from torch.export.exported_program import ExportedProgram
    from .TorchUtils import TorchUtils

def _normalize_example_inputs(example_inputs):
    # Accept torch.Tensor or sequence of tensors; always return a tuple of args
    if isinstance(example_inputs, torch.Tensor):
        return (example_inputs,)
    # If already a list/tuple
    if len(example_inputs) == 1 and isinstance(example_inputs[0], torch.Tensor):
        return (example_inputs[0],)
    return tuple(example_inputs)

def torchscript_to_dynamo(
    model: torch.ScriptModule, example_inputs
) -> ExportedProgram:
    """Converts a TorchScript module to a Dynamo program."""
    sample_args = _normalize_example_inputs(example_inputs)
    traced = torch.jit.trace(model, sample_args)
    return TS2EPConverter(traced, sample_args=sample_args, sample_kwargs=None).convert()

def nnmodule_to_dynamo(
    model: torch.nn.Module, example_inputs: list[torch.Tensor], dynamic_shapes=None
) -> ExportedProgram:
    """Converts a nn.Module to a Dynamo program."""
    return torch.export.export(
        model, tuple(example_inputs), dynamic_shapes=dynamic_shapes
    )

"""onnx_support = True
try:
    import onnx
except ImportError:
    onnx_support = False"""


class TorchTensorRTHandler: 
    """
    Args:
        trt_workspace_size (int): The workspace size to use when compiling models using TensorRT. Defaults to 0.
        max_aux_streams (int | None): The maximum number of auxiliary streams to use when compiling models using TensorRT. Defaults to None.
        trt_optimization_level (int): The optimization level to use when compiling models using TensorRT. Defaults to 3.
        debug (bool): Whether to enable debugging when compiling models using TensorRT. Defaults to False.


        multi precision engines seem to not like torchscript2exportedprogram,
        or maybe its just the model not playing nice with explicit_typing,
        either way, forcing one precision helps with speed in some cases.
    """

    trt_path_appendix = (f"_{__version__}.engine") # this is used to identify the models that were exported with this version of RVE

    def __init__(
        self,
        model_parent_path: str,
        max_aux_streams: int | None = None,
        debug: bool = False,
        trt_optimization_level: int = 3,
        trt_workspace_size: int = 0,
        
    ):
        self.tensorrt_version = trt.__version__  # can just grab version from here instead of importing trt and torch trt in all related files
        self.torch_tensorrt_version = torch_tensorrt.__version__

        self.trt_workspace_size = trt_workspace_size
        self.max_aux_streams = max_aux_streams
        self.optimization_level = trt_optimization_level
        self.debug = debug
        self.model_parent_path = model_parent_path
        # clear previous tensorrt models
        cleared_models = False
        if os.path.exists(self.model_parent_path):
            for model in os.listdir(self.model_parent_path):
                
                if not self.trt_path_appendix.lower() in model.lower() and "tensorrt" in model.lower():
                    model_path = os.path.join(self.model_parent_path, model)
                    try:
                        os.remove(model_path)
                        cleared_models = True
                        log(f"Removed {model_path}")
                    except Exception as e:
                        log(f"Failed to remove {model_path}: {e}")
            if cleared_models:
                print("Cleared old TensorRT models...", file=sys.stderr)
    
    

    def grid_sample_decomp(self, exported_program):
        from torch_tensorrt.dynamo.conversion.impl.grid import GridSamplerInterpolationMode
        GridSamplerInterpolationMode.update(
            {
                0: trt.InterpolationMode.LINEAR,
                1: trt.InterpolationMode.NEAREST,
                2: trt.InterpolationMode.CUBIC,
            }
        )
        return exported_program

    def check_engine_exists(self, trt_engine_name: str) -> bool:
        """Checks if a TensorRT engine exists at the specified path."""
        trt_engine_name += self.trt_path_appendix
        trt_engine_path = os.path.join(self.model_parent_path, trt_engine_name)
        return os.path.exists(trt_engine_path)

    def build_engine(
        self,
        model: torch.nn.Module,
        dtype: torch.dtype,
        device: torch.device,
        example_inputs: list[torch.Tensor],
        trt_engine_name: str,
        trt_multi_precision_engine: bool = False,
        dynamic_shapes: dict | None = None,
        
    ):
        """
        Returns a Torch TensorRT engine built from the provided model.
        """
        start_time = time.time()
        trt_engine_name += self.trt_path_appendix
        TorchUtils.clear_cache()
        """Builds a TensorRT engine from the provided model."""
        print(
            f"Building TensorRT engine {os.path.basename(trt_engine_name)}. This may take a while...",
            file=sys.stderr,
        )
        
        with suppress_stdout_stderr():
            exported_program = nnmodule_to_dynamo(model, example_inputs, dynamic_shapes=dynamic_shapes)
            TorchUtils.clear_cache()

            exported_program = self.grid_sample_decomp(exported_program)
            
            model_trt = torch_tensorrt.dynamo.compile(
                exported_program,
                tuple(example_inputs),
                device=device,
                enabled_precisions={dtype} if not trt_multi_precision_engine else {torch.float},
                use_explicit_typing=trt_multi_precision_engine,
                debug=self.debug,
                num_avg_timing_iters=4,
                workspace_size=self.trt_workspace_size,
                min_block_size=1,
                max_aux_streams=self.max_aux_streams,
                optimization_level=self.optimization_level,
                #tiling_optimization_level="full",
            )
        
        print(
            f"TensorRT engine built in {time.time() - start_time:.2f} seconds.",
            file=sys.stderr,
        )
        TorchUtils.clear_cache()
        return model_trt

    def save_engine(self, trt_engine: torch.jit.ScriptModule, trt_engine_name: str, example_inputs: list[torch.Tensor]):
        """Saves a TensorRT engine to the specified path."""
        trt_engine_name += self.trt_path_appendix
        trt_engine_path = os.path.join(self.model_parent_path, trt_engine_name)

        torch_tensorrt.save(
                trt_engine,
                trt_engine_path,
                output_format="torchscript",
                inputs=tuple(example_inputs),
        )
        TorchUtils.clear_cache()

    def load_engine(self, trt_engine_name: str) -> torch.jit.ScriptModule:
        """Loads a TensorRT engine from the specified path."""
        
        trt_engine_name += self.trt_path_appendix
        trt_engine_path = os.path.join(self.model_parent_path, trt_engine_name)
        print(f"Loading TensorRT engine from {trt_engine_path}.", file=sys.stderr)
        return torch.jit.load(trt_engine_path).eval()

