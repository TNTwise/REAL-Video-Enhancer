from __future__ import annotations

import os
from fractions import Fraction
from threading import Lock

import numpy as np
import tensorrt
import torch
from torch.autograd.function import InplaceFunction
import torch.nn.functional as F
import vapoursynth as vs
from torch_tensorrt.fx import LowerSetting
from torch_tensorrt.fx.lower import Lowerer
from torch_tensorrt.fx.utils import LowerPrecision
try:
    from src.programData.thisdir import thisdir
    thisdir = thisdir()
except:
    thisdir = os.getcwd()

class RifeTensorRT:
    def __init__(self,
                model: str = "rife414.pkl",
                width: int = 1920,
                height: int = 1080,
                scale: int = 1,
                ensemble: bool = False,
                precision: str = "fp16",
                trt_max_workspace_size: int = 1,
                num_streams: int = 2,
                ):

        self.width = width
        self.height = height
        self.scale = scale

        self.pad_frame()

        self.num_streams = num_streams
        self.ensemble = ensemble
        self.model = model
        self.device = torch.device("cuda")
        self.device_name = torch.cuda.get_device_name(self.device)
        self.trt_version = tensorrt.__version__
        self.dimensions = f"{self.pw}x{self.ph}"
        self.half = precision
        self.index = -1
        self.index_lock = Lock()
        self.stream = [torch.cuda.Stream(device=self.device) for _ in range(self.num_streams)]
        self.stream_lock = [Lock() for _ in range(self.num_streams)]
        self.trt_engine_path = os.path.join(
                    os.getcwd(),
                    (
                        f"{model}"
                        + f"_{self.device_name}"
                        + f"_trt-{self.trt_version}"
                        + f"_{self.dimensions}"
                        + f"_{self.half}"
                        + f"_workspace-{trt_max_workspace_size}"
                        + f"_scale-{scale}"
                        + f"_ensemble-{ensemble}"
                        + ".pt"
                    ),
                )

        if not os.path.exists(self.trt_engine_path):
            self.generateEngine()

        self.inference = [torch.load(self.trt_engine_path) for _ in range(self.num_streams)]

    def pad_frame(self):
        tmp = max(128, int(128 / self.scale))
        self.pw = ((self.width - 1) // tmp + 1) * tmp
        self.ph = ((self.height - 1) // tmp + 1) * tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)

    def generateEngine(self):
        # temp
        trt_max_workspace_size = 1
        fp16 = self.half
        scale = self.scale
        ensemble = self.ensemble
        model_name = self.model
        device = self.device
        w = self.width
        h = self.height

        if fp16:
                torch.set_default_dtype(torch.half)



        from src.torch.rife.rife46.IFNet_HDv3 import IFNet

        state_dict = torch.load(os.path.join(thisdir,"models","rife-cuda",model_name.replace(".",""), model_name+".pkl"), map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if "module." in k}

        flownet = IFNet()
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
        del flownet
        torch.cuda.empty_cache()

    def run1(self, I0, I1):
                self.I0 = self.frame_to_tensor(I0, self.device)
                self.I1 = self.frame_to_tensor(I1, self.device)
                self.I0 = F.pad(self.I0, self.padding)
                self.I1 = F.pad(self.I1, self.padding)

                if self.half:
                    self.I0 = self.I0.half()
                    self.I1 = self.I1.half()



    @torch.inference_mode()
    def make_inference(self, n):



        with self.index_lock:
            index = (self.index + 1) % self.num_streams
            local_index = index

        with self.stream_lock[local_index], torch.cuda.stream(self.stream[local_index]):
            timestep = torch.full((1, 1, self.I0.shape[2], self.I1.shape[3]), n, device=self.device)
            timestep = timestep.to(memory_format=torch.channels_last)
            if self.half:
                timestep = timestep.half()

            output = self.inference[local_index](self.I0, self.I1, timestep)
            output = output[:, :, : self.height, : self.width]
            output = (output[0] * 255.0).byte().cpu().numpy().transpose(1, 2, 0)

            return output

    def frame_to_tensor(self,frame, device: torch.device) -> torch.Tensor:
            array = frame
            return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device, memory_format=torch.channels_last)  / 255.0

if __name__ == '__main__':
    rifetrt = RifeTensorRT()
