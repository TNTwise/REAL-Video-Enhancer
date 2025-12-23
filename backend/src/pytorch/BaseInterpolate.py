import torch
from abc import ABCMeta, abstractmethod
from queue import Queue

from ..utils.SSIM import SSIM

# from backend.src.pytorch.InterpolateArchs.GIMM import GIMM
from .UpscaleTorch import UpscalePytorch
import logging
import gc
from ..utils.Util import CudaChecker

HAS_PYTORCH_CUDA = CudaChecker().HAS_PYTORCH_CUDA

torch.set_float32_matmul_precision("medium")
torch.set_grad_enabled(False)
logging.basicConfig(level=logging.INFO)


class DynamicScale:
    def __init__(self, possible_values: dict, CompareNet: SSIM):
        self.possible_values = possible_values
        self.CompareNet = CompareNet

    @torch.inference_mode()
    def dynamicScaleCalculation(self, frame0, frame1):
        ssim: torch.Tensor = self.CompareNet(frame0, frame1)
        closest_value = min(self.possible_values, key=lambda v: abs(ssim.item() - v))
        scale = self.possible_values[closest_value]
        return scale

    # limit gmfss scale to 1.0 max


class BaseInterpolate(metaclass=ABCMeta):
    @abstractmethod
    def _load(self):
        """Loads in the model"""
        self.HAS_PYTORCH_CUDA = HAS_PYTORCH_CUDA
        self.device = torch.device("cuda")
        self.dtype = torch.float32
        self.width = 1920
        self.height = 1080
        self.padding = [0, 0, 0, 0]
        self.frame0 = None
        self.encode0 = None
        self.flownet = None
        self.encode = None
        self.tenFlow_div = None
        self.backwarp_tenGrid = None
        self.doEncodingOnFrame = False  # set this by default
        self.hdr_mode = False
        self.CompareNet = None

    def hotUnload(self):
        self.flownet = None
        self.encode = None
        self.tenFlow_div = None
        self.backwarp_tenGrid = None
        self.f0encode = None
        gc.collect()
        if self.HAS_PYTORCH_CUDA:
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_max_memory_cached()

    @torch.inference_mode()
    def hotReload(self):
        self._load()

    @abstractmethod
    @torch.inference_mode()
    def __call__(
        self,
        img1,
        writeQueue: Queue,
        transition=False,
        upscaleModel: UpscalePytorch = None,
    ):  # type: ignore
        """Perform processing"""

    @torch.inference_mode()
    def uncacheFrame(self):
        self.f0encode = None 
        self.img0 = None
