import torch
import torch.nn.functional as F
from .TorchUtils import TorchUtils
# from backend.src.pytorch.InterpolateArchs.GIMM import GIMM
from .BaseInterpolate import BaseInterpolate, DynamicScale
import math
import logging
import sys
from ..utils.Util import (
    warnAndLog,
    log,
)
from ..utils.Util import CudaChecker
from ..utils.Frame import Frame
from time import sleep

torch.set_float32_matmul_precision("medium")
torch.set_grad_enabled(False)
logging.basicConfig(level=logging.INFO)

class InterpolateGMFSSTorch(BaseInterpolate):
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
        dynamicScaledOpticalFlow: bool = False,
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
        self.dynamicScaledOpticalFlow = dynamicScaledOpticalFlow
        self.UHDMode = UHDMode
        self.gpu_id = gpu_id

        self.CompareNet = None
        self.max_timestep = max_timestep
        if UHDMode:
            self.scale = 0.5
        _pad = 64
        if self.dynamicScaledOpticalFlow:
            tmp = max(_pad, int(_pad / 0.25))
        else:
            tmp = max(_pad, int(_pad / self.scale))
        self.pw = math.ceil(self.width / tmp) * tmp
        self.ph = math.ceil(self.height / tmp) * tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)
        self.device_type = device
        self.torchUtils = TorchUtils(
            self.width,
            self.height,
            hdr_mode=self.hdr_mode,
            device_type=device
        )
        self.device = self.torchUtils.handle_device(device, gpu_id=gpu_id)
        self.dtype = self.torchUtils.handle_precision(dtype)
        self._load()

    @torch.inference_mode()
    def _load(self):
        self.stream = self.torchUtils.init_stream(gpu_id=self.gpu_id)  
        self.prepareStream = self.torchUtils.init_stream(gpu_id=self.gpu_id)
        self.copy_stream = self.torchUtils.init_stream(gpu_id=self.gpu_id)
        with self.torchUtils.run_stream(self.prepareStream):  # type: ignore
            if self.dynamicScaledOpticalFlow:
                from ..utils.SSIM import SSIM

                compareNet = SSIM()
                self.CompareNet = compareNet.to(device=self.device, dtype=self.dtype)
                possible_values = {
                    0.25: 0.25,
                    0.5: 0.5,
                    0.75: 1.0,
                }  # closest_value:representative_scale
                self.dynamicScale = DynamicScale(
                    possible_values=possible_values, CompareNet=compareNet
                )
                print("Dynamic Scaled Optical Flow Enabled")
                if self.backend == "tensorrt":
                    print(
                        "Dynamic Scaled Optical Flow does not work with TensorRT, disabling",
                        file=sys.stderr,
                    )
                if self.UHDMode:
                    print(
                        "Dynamic Scaled Optical Flow does not work with UHD Mode, disabling",
                        file=sys.stderr,
                    )
            from .InterpolateArchs.GMFSS.GMFSS import GMFSS

            
            # caching the timestep tensor in a dict with the timestep as a float for the key
            self.timestepDict = {}
            for n in range(self.ceilInterpolateFactor):
                timestep = n / (self.ceilInterpolateFactor)
                timestep_tens = torch.tensor(
                    [timestep], dtype=self.dtype, device=self.device
                ).to(non_blocking=True)
                self.timestepDict[timestep] = timestep_tens
            self.flownet = GMFSS(
                model_path=self.interpolateModel,
                scale=self.scale,
                width=self.width,
                height=self.height,
                ensemble=self.ensemble,
                dtype=self.dtype,
                device=self.device,
                max_timestep=self.max_timestep,
            )

            log("GMFSS loaded")
            log("Scale: " + str(self.scale))
            HAS_SYSTEM_CUDA = CudaChecker().HAS_SYSTEM_CUDA
            log("Using System CUDA: " + str(HAS_SYSTEM_CUDA))
            if not HAS_SYSTEM_CUDA:
                print(
                    "WARNING: System CUDA not found, falling back to PyTorch softsplat. This will be a bit slower.",
                    file=sys.stderr,
                )
            if self.backend == "tensorrt":
                warnAndLog(
                    "TensorRT is not implemented for GMFSS yet, falling back to PyTorch"
                )
        self.torchUtils.sync_stream(self.prepareStream)  # type: ignore

    @torch.inference_mode()
    def __call__(
        self,
        img1: Frame,
        transition=False,
    ):  # type: ignore

        with self.torchUtils.run_stream(self.stream):  # type: ignore
            with self.torchUtils.run_stream(self.prepareStream):  # type: ignore
                if self.frame0 is None:
                    self.frame0 = F.pad(img1.get_frame_tensor(), self.padding)
                    return
            
                frame1 = F.pad(img1.get_frame_tensor(), self.padding)
            self.torchUtils.sync_stream(self.prepareStream)
            
            if self.dynamicScaledOpticalFlow:
                closest_value = self.dynamicScale.dynamicScaleCalculation(
                    self.frame0, frame1
                )
            else:
                closest_value = None

            for n in range(self.ceilInterpolateFactor - 1):
                if not transition:
                    timestep = (n + 1) * 1.0 / (self.ceilInterpolateFactor)
                    while self.flownet is None:
                        sleep(1)
                    timestep = self.timestepDict[timestep]

                    yield (
                        img1
                        .get_dummy_frame()
                        .set_frame_tensor(
                            self.flownet.forward(self.frame0, frame1, timestep, closest_value)
                            [:, :, : self.height, : self.width]
                            )
                        )
                else:
                    self.flownet.reset_cache_after_transition()
                    yield img1


            self.torchUtils.copy_tensor(self.frame0, frame1, self.copy_stream)

        self.torchUtils.sync_all_streams()
