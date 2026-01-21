

# from backend.src.pytorch.InterpolateArchs.GIMM import GIMM
from .InterpolateArchs.DetectInterpolateArch import ArchDetect
from .InterpolateGIMM import InterpolateGIMMTorch
from .InterpolateGMFSS import InterpolateGMFSSTorch
from .InterpolateIFRNET import InterpolateIFRNetTorch
from .InterpolateRIFE import InterpolateRIFEDRBA, InterpolateRifeTorch


class InterpolateFactory:
    @staticmethod
    def build_interpolation_method(interpolate_model_path, backend, drba=False):
        ad = ArchDetect(interpolate_model_path)
        base_arch = ad.getArchBase()
        match base_arch:
            case 'rife':
                if drba:
                    return InterpolateRIFEDRBA
                return InterpolateRifeTorch
            case 'gimm':
                return InterpolateGIMMTorch
            case 'gmfss':
                return InterpolateGMFSSTorch
            case 'ifrnet':
                return InterpolateIFRNetTorch  # IFRNet is a RIFE based architecture
