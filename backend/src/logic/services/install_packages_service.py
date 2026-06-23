import asyncio
import sys

from src.logic.proxy.settings import PersistentSettingsProxy
from src.utils.LogConfig import get_logger

logger = get_logger(__name__)


class InstallPackagesService:
    def __init__(self, settings_proxy: PersistentSettingsProxy):
        self._settings_proxy = settings_proxy
        self._current_process: asyncio.subprocess.Process | None = None
        self._current_stdout: str = ""
        self._current_status: str = "idle"

    @property
    def current_stdout(self) -> str:
        return self._current_stdout

    @property
    def current_status(self) -> str:
        return self._current_status

    async def _install_packages(self, packages: list[str], upgrade: bool = False):
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-warn-script-location",
            "--isolated",
            "--extra-index-url",
            "https://download.pytorch.org/whl/",
            "--trusted-host",
            "download.pytorch.org",
        ]
        if upgrade:
            cmd.append("--upgrade")
        cmd.extend(packages)

        self._current_stdout = ""
        self._current_status = "installing"
        self._current_process = None

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self._current_process = process

        while True:
            line = await process.stdout.readline()
            if not line:
                break
            decoded = line.decode(errors="replace")
            self._current_stdout += decoded

        await process.wait()

        if process.returncode == 0:
            self._current_status = "done"
            logger.info(f"Successfully installed: {', '.join(packages)}")
        else:
            self._current_status = "error"
            logger.error(f"Installation failed:\n{self._current_stdout}")

    async def install_base_packages(self):
        base_packages = [
            "requests",
            "opencv-python-headless",
            "numpy",
            "typing_extensioons",
        ]
        await self._install_packages(base_packages)

    async def install_torch_packages(self, upgrade: bool = False):
        torch_version = f"torch=={self._settings_proxy.pytorch_version}+{self._settings_proxy.torch_accelerator}"
        torchvision_version = f"torchvision=={self._settings_proxy.torchvision_version}+{self._settings_proxy.torch_accelerator}"
        await self._install_packages(
            [torch_version, torchvision_version], upgrade=upgrade
        )

    async def install_ncnn_packages(self, upgrade: bool = False):
        ncnn_packages = ["ncnn", "rife-ncnn-vulkan-python-tntwise", "upscale_ncnn_py"]
        await self._install_packages(ncnn_packages, upgrade=upgrade)

    async def install_tensorrt_packages(self, upgrade: bool = False):
        tensorrt_version = f"tensorrt=={self._settings_proxy.tensorrt_version}"
        torch_tensorrt = f"torch_tensorrt=={self._settings_proxy.pytorch_version}"
        await self._install_packages(
            [tensorrt_version, torch_tensorrt], upgrade=upgrade
        )
