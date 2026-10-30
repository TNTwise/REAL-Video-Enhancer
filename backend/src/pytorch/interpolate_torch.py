import torch
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod
from queue import Queue

from ..utils.ssim import SSIM

# from backend.src.pytorch.InterpolateArchs.GIMM import GIMM
from .InterpolateArchs.detect_interpolate_arch import ArchDetect
from .interpolate_gmfss import InterpolateGMFSSTorch
from .interpolate_gimm import InterpolateGIMMTorch
from .interpolate_rife import InterpolateRifeTorch
from .interpolate_ifrnet import InterpolateIFRNetTorch


class InterpolateFactory:
    @staticmethod
    def build_interpolation_method(interpolate_model_path, backend):
        ad = ArchDetect(interpolate_model_path)
        base_arch = ad.getArchBase()
        match base_arch:
            case "rife":
                return InterpolateRifeTorch
            case "gimm":
                return InterpolateGIMMTorch
            case "gmfss":
                return InterpolateGMFSSTorch
            case "ifrnet":
                return InterpolateIFRNetTorch  # IFRNet is a RIFE based architecture

    @staticmethod
    def build_interpolator(
        interpolate_model_path: str,
        backend: str = "pytorch",
        modelPath: str = None,
        ceilInterpolateFactor: int = 2,
        width: int = 1920,
        height: int = 1080,
        device: str = "default",
        dtype: str = "auto",
        gpu_id: int = 0,
        UHDMode: bool = False,
        ensemble: bool = False,
        dynamicScaledOpticalFlow: bool = False,
        max_timestep: float = 1,
        hdr_mode: bool = False,
        trt_optimization_level: int = 3,
        trt_static_shape: bool = False,
    ):
        """
        Builds the interpolation implementation used to interpolate frames for the given backend/model.

        Args:
            interpolate_model_path (str): path to the interpolation model.
            backend (str, optional): backend used for inference (pytorch/ncnn/tensorrt). Defaults to "pytorch".
            modelPath (str, optional): path to the interpolation model, defaults to interpolate_model_path.
            ceilInterpolateFactor (int, optional): rounded up interpolation factor. Defaults to 2.
            width (int, optional): width of the frame being interpolated. Defaults to 1920.
            height (int, optional): height of the frame being interpolated. Defaults to 1080.
            device (str, optional): device used for inference. Defaults to "default".
            dtype (str, optional): precision used for inference. Defaults to "auto".
            gpu_id (int, optional): gpu id used for inference. Defaults to 0.
            UHDMode (bool, optional): whether UHD mode is used. Defaults to False.
            ensemble (bool, optional): whether to use the ensemble. Defaults to False.
            dynamicScaledOpticalFlow (bool, optional): whether to use dynamic scaled optical flow. Defaults to False.
            max_timestep (float, optional): max timestep used for scene change caching. Defaults to 1.
            hdr_mode (bool, optional): whether the frames are hdr. Defaults to False.
            trt_optimization_level (int, optional): tensorrt optimization level. Defaults to 3.
            trt_static_shape (bool, optional): whether the tensorrt engine uses static shapes. Defaults to False.

        Returns:
            The constructed interpolation implementation, with __call__.
        """
        interpolate_class = InterpolateFactory.build_interpolation_method(
            interpolate_model_path, backend
        )

        if backend == "ncnn":
            from ..ncnn.interpolate_ncnn import InterpolateRIFENCNN

            return InterpolateRIFENCNN(
                interpolateModelPath=interpolate_model_path,
                width=width,
                height=height,
                gpuid=gpu_id,
                max_timestep=max_timestep,
                interpolateFactor=ceilInterpolateFactor,
                hdr_mode=hdr_mode,
            )

        return interpolate_class(
            modelPath=modelPath if modelPath is not None else interpolate_model_path,
            ceilInterpolateFactor=ceilInterpolateFactor,
            width=width,
            height=height,
            device=device,
            dtype=dtype,
            backend=backend,
            gpu_id=gpu_id,
            UHDMode=UHDMode,
            ensemble=ensemble,
            dynamicScaledOpticalFlow=dynamicScaledOpticalFlow,
            trt_optimization_level=trt_optimization_level,
            hdr_mode=hdr_mode,
            trt_static_shape=trt_static_shape,
            max_timestep=max_timestep,
        )
