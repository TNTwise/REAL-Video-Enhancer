import platform

from fastapi import APIRouter

from src.logic.handlers.backend.pytorch_handler import TorchHandler
from src.schemas.domain.system import GPUInfo, SystemInfo
from src.version import __version__

router = APIRouter(prefix="/system", tags=["System"])


def _get_opencv_version() -> str | None:
    try:
        import cv2

        return cv2.__version__
    except ImportError:
        return None


def _get_total_memory_gb() -> float | None:
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / 1_048_576, 1)
        elif platform.system() == "Darwin":
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
            )
            if result.returncode == 0:
                return round(int(result.stdout.strip()) / 1_073_741_824, 1)
        elif platform.system() == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            memory_status = MEMORYSTATUSEX()
            memory_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status)):
                return round(memory_status.ullTotalPhys / 1_073_741_824, 1)
    except Exception:
        pass
    return None


def _get_cuda_version(torch_module) -> str | None:
    try:
        if torch_module.cuda.is_available():
            ver = torch_module.cuda_version
            return f"cu{ver}" if ver else "cuda"
        return None
    except Exception:
        return None


def _detect_gpus() -> list[GPUInfo]:
    try:
        from gpu_detector import detect

        detected = detect()
        return [
            GPUInfo(
                name=gpu.name,
                vendor=gpu.vendor,
                memory_mb=gpu.memory_mb,
                driver_version=gpu.driver_version,
                device_id=gpu.device_id,
            )
            for gpu in detected
        ]
    except Exception:
        return []


@router.get("/info", response_model=SystemInfo)
def get_system_info():
    torch_handler = TorchHandler()
    pytorch_version: str | None = None
    torch_accelerator = "cpu"
    cuda_version: str | None = None

    if torch_handler.is_available():
        torch_module = torch_handler.get_torch()
        pytorch_version = torch_module.__version__
        if torch_module.cuda.is_available():
            torch_accelerator = "cuda"
            cuda_version = _get_cuda_version(torch_module)

    return SystemInfo(
        python_version=platform.python_version(),
        app_version=__version__,
        opencv_version=_get_opencv_version(),
        pytorch_version=pytorch_version,
        cuda_version=cuda_version,
        torch_accelerator=torch_accelerator,
        total_memory_gb=_get_total_memory_gb(),
        gpus=_detect_gpus(),
    )
