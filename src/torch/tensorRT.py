from __future__ import annotations

import os
from fractions import Fraction
from threading import Lock

import numpy as np
import tensorrt
import torch
import torch.nn.functional as F
import vapoursynth as vs
from torch_tensorrt.fx import LowerSetting
from torch_tensorrt.fx.lower import Lowerer
from torch_tensorrt.fx.utils import LowerPrecision

class RifeTensorRT:
    def __init__(self,
                model: str = "rife414.pkl",
                width: int = 1920,
                height: int = 1080,
                scale: int = 1,
                ensemble: bool = False,
                precision: str = "fp16",
                trt_max_workspace_size: int = 1,
                ):

        self.width = width
        self.height = height
        self.scale = scale

        self.padding()

        self.ensemble = ensemble
        self.model = model
        self.device = torch.device("cuda")
        self.device_name = torch.cuda.get_device_name(self.device)
        self.trt_version = tensorrt.__version__
        self.dimensions = f"{self.pw}x{self.ph}"
        self.precision = precision
        self.trt_engine_path = os.path.join(
                    os.getcwd(),
                    (
                        f"{model}"
                        + f"_{self.device_name}"
                        + f"_trt-{self.trt_version}"
                        + f"_{self.dimensions}"
                        + f"_{self.precision}"
                        + f"_workspace-{trt_max_workspace_size}"
                        + f"_scale-{scale}"
                        + f"_ensemble-{ensemble}"
                        + ".pt"
                    ),
                )
        if not os.path.exists(self.trt_engine_path):
            self.generateEngine()

    def padding(self):
        tmp = max(128, int(128 / self.scale))
        self.pw = ((self.width - 1) // tmp + 1) * tmp
        self.ph = ((self.height - 1) // tmp + 1) * tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)

    def generateEngine(self):
        # temp
        trt_max_workspace_size = 1
        fp16 = self.precision
        scale = self.scale
        ensemble = self.ensemble
        model_name = self.model
        device = self.device
        w = self.width
        h = self.height

        if fp16:
                torch.set_default_dtype(torch.half)





        from rife.rife414.IFNet_HDv3 import IFNet
        state_dict = torch.load(os.path.join("rife",model_name.replace(".pkl",""), model_name), map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if "module." in k}

        flownet = IFNet(scale, ensemble)
        flownet.load_state_dict(state_dict, strict=False)
        flownet.eval().to(device, memory_format=torch.channels_last)


        lower_setting = LowerSetting(
            lower_precision=LowerPrecision.FP16 if fp16 else LowerPrecision.FP32,
            min_acc_module_size=1,
            max_workspace_size=trt_max_workspace_size,
            dynamic_batch=False,
            tactic_sources=1 << int(tensorrt.TacticSource.EDGE_MASK_CONVOLUTIONS)
            | 1 << int(tensorrt.TacticSource.JIT_CONVOLUTIONS),
        )
        lowerer = Lowerer.create(lower_setting=lower_setting)
        flownet = lowerer(
            flownet,
            [
                torch.zeros((1, 3, self.ph, self.pw), device=device).to(memory_format=torch.channels_last),
                torch.zeros((1, 3, self.ph, self.pw), device=device).to(memory_format=torch.channels_last),
                torch.zeros((1, 1, self.ph, self.pw), device=device).to(memory_format=torch.channels_last),
            ],
        )
        torch.save(flownet, self.trt_engine_path)

rifetrt = RifeTensorRT()
