import torch
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod
from queue import Queue

from ..utils.SSIM import SSIM

# from backend.src.pytorch.InterpolateArchs.GIMM import GIMM
from .InterpolateArchs.DetectInterpolateArch import ArchDetect
from .InterpolateGMFSS import InterpolateGMFSSTorch
from .InterpolateGIMM import InterpolateGIMMTorch
from .InterpolateRIFE import InterpolateRifeTorch,  InterpolateRIFEDRBA
from .InterpolateIFRNET import InterpolateIFRNetTorch


class InterpolateFactory:
    @staticmethod
    def build_interpolation_method(interpolate_model_path, backend, drba=False):
        ad = ArchDetect(interpolate_model_path)
        base_arch = ad.getArchBase()
        match base_arch:
            case "rife":
                if drba:
                    return InterpolateRIFEDRBA
                return InterpolateRifeTorch
            case "gimm":
                return InterpolateGIMMTorch
            case "gmfss":
                return InterpolateGMFSSTorch
            case "ifrnet":
                return InterpolateIFRNetTorch  # IFRNet is a RIFE based architecture
