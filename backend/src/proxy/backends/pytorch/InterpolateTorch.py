from src.schemas.domain.model import InterpolateModel
from src.schemas.domain.render import RenderSettings
from src.schemas.domain.video_info import InputVideoInfo

from .InterpolateArchs.DetectInterpolateArch import ArchDetect
from .InterpolateGIMM import InterpolateGIMMTorch
from .InterpolateGMFSS import InterpolateGMFSSTorch
from .InterpolateIFRNET import InterpolateIFRNetTorch
from .InterpolateRIFE import InterpolateRifeTorch


class InterpolateFactory:
    @staticmethod
    def build_interpolation_method(
        interpolate_model: InterpolateModel,
        video_info: InputVideoInfo,
        render_settings: RenderSettings,
        settings,
    ):
        ad = ArchDetect(interpolate_model.file_path)
        base_arch = ad.getArchBase()
        match base_arch:
            case "rife":
                return InterpolateRifeTorch(
                    interpolate_model=interpolate_model,
                    video_info=video_info,
                    render_settings=render_settings,
                    settings=settings,
                )
            case "gimm":
                return InterpolateGIMMTorch(
                    modelPath=interpolate_model.file_path,
                    ceilInterpolateFactor=interpolate_model.interpolate_factor,
                    width=video_info.width,
                    height=video_info.height,
                    device="default",
                    dtype=settings.precision,
                    backend=interpolate_model.backend.type,
                    UHDMode=settings.uhd_mode == "True",
                    hdr_mode=video_info.is_hdr,
                    gpu_id=int(settings.pytorch_gpu_id),
                )
            case "gmfss":
                return InterpolateGMFSSTorch(
                    modelPath=interpolate_model.file_path,
                    ceilInterpolateFactor=interpolate_model.interpolate_factor,
                    width=video_info.width,
                    height=video_info.height,
                    device="default",
                    dtype=settings.precision,
                    backend=interpolate_model.backend.type,
                    UHDMode=settings.uhd_mode == "True",
                    hdr_mode=video_info.is_hdr,
                    gpu_id=int(settings.pytorch_gpu_id),
                )
            case "ifrnet":
                return InterpolateIFRNetTorch(
                    modelPath=interpolate_model.file_path,
                    ceilInterpolateFactor=interpolate_model.interpolate_factor,
                    width=video_info.width,
                    height=video_info.height,
                    device="default",
                    dtype=settings.precision,
                    backend=interpolate_model.backend.type,
                    UHDMode=settings.uhd_mode == "True",
                    hdr_mode=video_info.is_hdr,
                    gpu_id=int(settings.pytorch_gpu_id),
                )
