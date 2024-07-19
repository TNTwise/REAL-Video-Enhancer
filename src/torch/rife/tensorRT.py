from __future__ import annotations

import os
import math

try:
    import tensorrt

    import torch
    from torch.autograd.function import InplaceFunction
    import torch.nn.functional as F
    import torch_tensorrt
except:
    pass
torch.set_float32_matmul_precision("medium")

try:
    from src.programData.thisdir import thisdir

    thisdir = thisdir()
except:
    thisdir = os.getcwd()

from threading import Lock


class RifeTensorRT:
    @torch.inference_mode()
    def __init__(
        self,
        model: str = "rife414.pkl",
        width: int = 1920,
        height: int = 1080,
        scale: int = 1,
        ensemble: bool = False,
        half: bool = False,
        trt_max_workspace_size: int = 1,
        num_streams: int = 1,
        guiLog=None,
        device_index: int = 0,
        trt_min_shape: list[int] = [128, 128],
        trt_opt_shape: list[int] = [1920, 1080],
        trt_max_shape: list[int] = [1920, 1080],
    ):
        self.width = width
        self.height = height
        self.scale = scale

        # padding
        self.tmp = max(32, int(32 / scale))
        self.pw = math.ceil(self.width / self.tmp) * self.tmp
        self.ph = math.ceil(self.height / self.tmp) * self.tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)

        self.num_streams = num_streams
        self.ensemble = ensemble
        self.model = model
        self.device = torch.device("cuda", device_index)
        self.device_name = torch.cuda.get_device_name(self.device)
        self.trt_version = tensorrt.__version__
        self.guiLog = guiLog
        self.half = half
        self.dtype = torch.half if self.half else torch.float
        self.trt_optimization_level = 5
        self.trt_workspace_size = 0
        self.trt_max_aux_streams = None
        self.torch_trt_version = torch_tensorrt.__version__
        self.trt_min_shape = trt_min_shape
        self.trt_max_shape = trt_max_shape
        self.trt_opt_shape = trt_opt_shape
        self.tenFlow_div = torch.tensor(
            [(self.pw - 1.0) / 2.0, (self.ph - 1.0) / 2.0],
            dtype=self.dtype,
            device=self.device,
        )
        self.handle_model(model)

        tenHorizontal = (
            torch.linspace(-1.0, 1.0, self.pw, dtype=self.dtype, device=self.device)
            .view(1, 1, 1, self.pw)
            .expand(-1, -1, self.ph, -1)
        )
        tenVertical = (
            torch.linspace(-1.0, 1.0, self.ph, dtype=self.dtype, device=self.device)
            .view(1, 1, self.ph, 1)
            .expand(-1, -1, -1, self.pw)
        )
        self.backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1)
        for i in range(2):
            self.trt_min_shape[i] = (
                math.ceil(max(self.trt_min_shape[i], 1) / self.tmp) * self.tmp
            )
            self.trt_opt_shape[i] = (
                math.ceil(max(self.trt_opt_shape[i], 1) / self.tmp) * self.tmp
            )
            self.trt_max_shape[i] = (
                math.ceil(max(self.trt_max_shape[i], 1) / self.tmp) * self.tmp
            )

        dimensions = (
            f"min-{self.trt_min_shape[0]}x{self.trt_min_shape[1]}"
            f"_opt-{self.trt_opt_shape[0]}x{self.trt_opt_shape[1]}"
            f"_max-{self.trt_max_shape[0]}x{self.trt_max_shape[1]}"
        )
        self.trt_engine_path = os.path.join(
            thisdir,
            "models",
            "rife-trt-engines",
            (
                f"{model}"
                + f"_{self.device_name}"
                + f"_trt-{self.trt_version}"
                + f"_torch_trt-{self.torch_trt_version}"
                + f"_{dimensions}"
                + f"_{self.half}"
                + f"_workspace-{trt_max_workspace_size}"
                + f"_scale-{scale}"
                + f"_ensemble-{ensemble}"
                + (
                    f"_workspace-{self.trt_workspace_size}"
                    if self.trt_workspace_size > 0
                    else ""
                )
                + (
                    f"_aux-{self.trt_max_aux_streams}"
                    if self.trt_max_aux_streams is not None
                    else ""
                )
                 
                + ".ts"
            ),
        )

        if not os.path.exists(self.trt_engine_path):
            self.generateEngine()

        self.inference = torch.jit.load(self.trt_engine_path)

    def handle_model(self, interpolate_method):
        if interpolate_method == "rife4.14":
            from .rife414.IFNet_HDv3 import IFNet

            modelDir = os.path.dirname(
                os.path.join(
                    f"{thisdir}", "models", "rife-cuda", "rife414", "rife4.14.pkl"
                )
            )

        if interpolate_method == "rife4.6":
            from .rife46.IFNet_HDv3 import IFNet

            modelDir = os.path.dirname(
                os.path.join(
                    f"{thisdir}", "models", "rife-cuda", "rife46", "rife4.6.pkl"
                )
            )
        if interpolate_method == "rife4.13-lite":
            from .rife413lite.IFNet_HDv3 import IFNet

            modelDir = os.path.dirname(
                os.path.join(
                    f"{thisdir}",
                    "models",
                    "rife-cuda",
                    "rife413-lite",
                    "rife4.13-lite.pkl",
                )
            )
        if interpolate_method == "rife4.14-lite":
            from .rife414lite.IFNet_HDv3 import IFNet

            modelDir = os.path.dirname(
                os.path.join(
                    f"{thisdir}",
                    "models",
                    "rife-cuda",
                    "rife414-lite",
                    "rife4.14-lite.pkl",
                )
            )

        if interpolate_method == "rife4.15":
            from .rife415.IFNet_HDv3 import IFNet

            modelDir = os.path.dirname(
                os.path.join(
                    f"{thisdir}", "models", "rife-cuda", "rife415", "rife4.15.pkl"
                )
            )
        if interpolate_method == "rife4.16-lite":
            from .rife416lite.IFNet_HDv3 import IFNet

            modelDir = os.path.dirname(
                os.path.join(
                    f"{thisdir}",
                    "models",
                    "rife-cuda",
                    "rife416-lite",
                    "rife4.16-lite.pkl",
                )
            )
        if interpolate_method == "rife4.17":
            from .rife415.IFNet_HDv3 import IFNet

            modelDir = os.path.dirname(
                os.path.join(
                    f"{thisdir}", "models", "rife-cuda", "rife417", "rife4.17.pkl"
                )
            )
        self.i = IFNet(scale=self.scale, ensemble=self.ensemble).to(device=self.device,dtype=self.dtype)
        self.i.cuda().half()
        self.modelDir = modelDir

    @torch.inference_mode()
    def generateEngine(self):
        inputs = [
                    torch.zeros(
                        (1, 3, self.ph, self.pw), dtype=self.dtype, device=self.device
                    ),
                    torch.zeros(
                        (1, 3, self.ph, self.pw), dtype=self.dtype, device=self.device
                    ),
                    torch.zeros(
                        (1, 1, self.ph, self.pw), dtype=self.dtype, device=self.device
                    ),
                    torch.zeros((2,),dtype=self.dtype,device=self.device),
                    torch.zeros(
                        (1, 2, self.ph, self.pw), dtype=self.dtype, device=self.device
                    )
                ]
        model = self.i
        module = torch.jit.trace(model, inputs)
        module = torch_tensorrt.compile(
                    module,
                    ir="ts",
                    inputs=inputs,
                    enabled_precisions={self.dtype},
                    device=torch_tensorrt.Device(gpu_id=0),
                    workspace_size=self.trt_workspace_size,
                    truncate_long_and_double=True,
                    min_block_size=1,
                )
        torch.jit.save(module, self.trt_engine_path)
        # flownet = [torch.export.load(self.trt_engine_path).module() for _ in range(self.num_streams)]

        # self.index = -1
        # self.index_lock = Lock()

    def pad_frame(self):
        self.I0 = F.pad(self.I0, [0, self.padding[1], 0, self.padding[3]])
        self.I1 = F.pad(self.I1, [0, self.padding[1], 0, self.padding[3]])

    @torch.inference_mode()
    def run1(self, I0, I1):
        self.I0 = self.frame_to_tensor(I0, self.device)
        self.I1 = self.frame_to_tensor(I1, self.device)

        if self.half:
            self.I0 = self.I0.half()
            self.I1 = self.I1.half()

        if self.padding != (0, 0, 0, 0):
            self.I0 = F.pad(self.I0, self.padding)
            self.I1 = F.pad(self.I1, self.padding)

    @torch.inference_mode()
    def make_inference(self, n):
        timestep = torch.full(
            (1, 1, self.ph, self.pw), n, dtype=self.dtype, device=self.device
        )

        output = self.inference(
            self.I0, self.I1, timestep, self.tenFlow_div, self.backwarp_tenGrid
        )
        output = output[:, :, : self.height, : self.width]

        return (
            (output[0])
            .squeeze(0)
            .permute(1, 2, 0)
            .mul(255.0)
            .byte()
            .contiguous()
            .cpu()
            .numpy()
        )

    @torch.inference_mode()
    def frame_to_tensor(self, frame, device: torch.device) -> torch.Tensor:
        frame = torch.frombuffer(frame, dtype=torch.uint8).reshape(
            self.height, self.width, 3
        )
        return (frame).permute(2, 0, 1).unsqueeze(0).to(
            device,
        ) / 255.0


if __name__ == "__main__":
    rifetrt = RifeTensorRT()
