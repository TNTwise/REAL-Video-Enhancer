import torch
from .TorchUtils import TorchUtils
# from backend.src.pytorch.InterpolateArchs.GIMM import GIMM
from .BaseInterpolate import BaseInterpolate
import math
import logging
from ..utils.Util import (
    warnAndLog,
    log,
)
from ..utils.Frame import Frame
from typing import Generator
torch.set_float32_matmul_precision("medium")
torch.set_grad_enabled(False)
logging.basicConfig(level=logging.INFO)

class InterpolateIFRNetTorch(BaseInterpolate):
    @torch.inference_mode()
    def __init__(
        self,
        modelPath: str,
        ceilInterpolateFactor: int = 2,
        width: int = 1920,
        height: int = 1080,
        device: str = "default",
        dtype: str = "auto",
        backend: str = "pytorch",
        UHDMode: bool = False,
        ensemble: bool = False,
        gpu_id: int = 0,
        max_timestep: float = 1,
        hdr_mode: bool = False,
        *args,
        **kwargs,
    ):
        self.frame0 = None
        self.interpolateModel = modelPath
        self.width = width
        self.height = height
        
        self.backend = backend
        self.ceilInterpolateFactor = ceilInterpolateFactor
        # set up streams for async processing
        self.scale = 1
        self.ensemble = ensemble
        self.hdr_mode = hdr_mode # used in base interpolate class (ik inheritance is bad leave me alone)
        self.UHDMode = UHDMode
        self.gpu_id = gpu_id
        self.CompareNet = None
        self.max_timestep = max_timestep
        if UHDMode:
            self.scale = 0.5
        _pad = 32
        tmp = max(_pad, int(_pad / self.scale))
        self.pw = math.ceil(self.width / tmp) * tmp
        self.ph = math.ceil(self.height / tmp) * tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)
        self.torchUtils = TorchUtils(
            self.width,
            self.height,
            hdr_mode=self.hdr_mode,
            device_type=device
        )
        self.device = self.torchUtils.handle_device(device, gpu_id=gpu_id)
        self.dtype = self.torchUtils.handle_precision(dtype)
        self.tenFlow_div = torch.tensor(
        [(self.pw - 1.0) / 2.0, (self.ph - 1.0) / 2.0],
            dtype=torch.float32,
            device=self.device,
        )
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, self.pw, dtype=torch.float32, device=self.device)
            .view(1, 1, 1, self.pw)
            .expand(-1, -1, self.ph, -1)
        ).to(dtype=torch.float32, device=self.device)
        tenVertical = (
            torch.linspace(-1.0, 1.0, self.ph, dtype=torch.float32, device=self.device)
            .view(1, 1, self.ph, 1)
            .expand(-1, -1, -1, self.pw)
        ).to(dtype=torch.float32, device=self.device)
        self.backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1)
        self._load()

    @torch.inference_mode()
    def _load(self):
        self.stream = self.torchUtils.init_stream(self.gpu_id)
        self.prepareStream = self.torchUtils.init_stream(self.gpu_id)
        with self.torchUtils.run_stream(self.prepareStream):  # type: ignore
            
            from .InterpolateArchs.IFRNET.IFRNet import IFRNet

            
            # caching the timestep tensor in a dict with the timestep as a float for the key
            timesteplist = []
            for n in range(self.ceilInterpolateFactor-1):
                timestep_tens = torch.tensor(
                    (n+1) / (self.ceilInterpolateFactor), dtype=self.dtype, device=self.device
                ).view(1, 1, 1, 1).to(non_blocking=True)
                timesteplist.append(timestep_tens)
            self.timestep = torch.cat(timesteplist, dim=0)
            
            self.flownet = IFRNet(
                scale_factor=self.scale,
            )
                
            state_dict = torch.load(
                self.interpolateModel,
                map_location="cpu",
                weights_only=True,
            )
            self.flownet.load_state_dict(
                state_dict,
                strict=True,
            )
            self.flownet.eval().to(device=self.device, dtype=self.dtype)
            log("IFRNet loaded")
            log("Scale: " + str(self.scale))
            
            if self.backend == "tensorrt":
                warnAndLog(
                    "TensorRT is not implemented for IFRNet yet, falling back to PyTorch"
                )
        self.torchUtils.sync_stream(self.prepareStream)  # type: ignore

    @torch.inference_mode()
    def __call__(
        self,
        img1: Frame,
        transition=False,
    ) -> Generator[Frame, Frame, Frame]:  # type: ignore

        with self.torchUtils.run_stream(self.stream):  # type: ignore
            with self.torchUtils.run_stream(self.prepareStream):  # type: ignore
                if self.frame0 is None:
                    self.frame0 = torch.nn.functional.pad(img1.get_frame_tensor(), self.padding)
                    self.frame0 = torch.cat([self.frame0 for _ in range(self.ceilInterpolateFactor-1)], dim=0)
                    return
                frame1 = torch.nn.functional.pad(img1.get_frame_tensor(), self.padding)
                frame1 = torch.cat([frame1 for _ in range(self.ceilInterpolateFactor-1)], dim=0)
            self.torchUtils.sync_stream(self.prepareStream)  # type: ignore
            
            if transition:
                for n in range(self.ceilInterpolateFactor - 1):
                    yield img1
            frames = self.flownet( # idk why but i gotta inference the frames every time, or else transitions will get cooked on higher interps
                self.frame0,
                frame1,
                self.timestep,
            )
            for frame in frames: 
                if not transition:
                    yield img1.get_dummy_frame().set_frame_tensor(frame.unsqueeze(0)[:, :, :self.height, :self.width])
            self.torchUtils.copy_tensor(self.frame0, frame1, self.prepareStream)

        self.torchUtils.sync_all_streams()
