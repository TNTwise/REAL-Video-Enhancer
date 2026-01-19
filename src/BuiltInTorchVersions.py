import requests

try:
    from .constants import HAS_NETWORK_ON_STARTUP
except ImportError:
    from constants import HAS_NETWORK_ON_STARTUP
from dataclasses import dataclass
from .ui.SettingsTab import Settings


@dataclass
class TorchVersion:
    torch_version: str
    torchvision_version: str
    cuda_version: str
    rocm_version: str
    xpu_version: str
    mps_version: str


class Torch2_9(TorchVersion):
    torch_version = "2.9.0"
    torchvision_version = "0.24.0"
    cuda_version = "+cu130"
    rocm_version = "+rocm6.4"
    xpu_version = "+xpu"
    mps_version = ""


class Torch2_8(TorchVersion):
    torch_version = "2.8.0"
    torchvision_version = "0.23.0"
    cuda_version = "+cu129"
    rocm_version = "+rocm6.4"
    xpu_version = "+xpu"
    mps_version = ""


class Torch2_6(TorchVersion):
    torch_version = "2.6.0"
    torchvision_version = "0.21.0"
    cuda_version = "+cu118"
    rocm_version = "+rocm6.2.4"
    xpu_version = "+xpu"
    mps_version = ""
