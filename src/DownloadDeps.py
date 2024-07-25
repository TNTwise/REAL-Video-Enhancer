from .Util import (
    getPlatform,
    printAndLog,
    pythonPath,
    ffmpegPath,
    currentDirectory,
    createDirectory,
    removeFile,
    makeExecutable,
    move,
)
from .QTcustom import DownloadProgressPopup, DisplayCommandOutputPopup
import os
import tarfile
import subprocess


class DownloadDependencies:
    """
    Downloads platform specific dependencies python and ffmpeg to their respective locations and creates the directories

    """

    def __init__(self):
        createDirectory(os.path.join(currentDirectory(), "python"))
        createDirectory(os.path.join(currentDirectory(), "ffmpeg"))

    def extractTarGZ(self, file):
        """
        Extracts a tar gz in the same directory as the tar file and deleted it after extraction.
        """
        printAndLog("Extracting: " + file)
        tar = tarfile.open(file, "r:gz")
        tar.extractall()
        tar.close()
        removeFile(file)

    def downloadPython(self):
        link = "https://github.com/indygreg/python-build-standalone/releases/download/20240713/cpython-3.11.9+20240713-"
        pyDir = os.path.join(
            currentDirectory(),
            "python",
            "python.tar.gz",
        )
        match getPlatform():
            case "linux":
                link += "x86_64-unknown-linux-gnu-install_only.tar.gz"
            case "win32":
                link += "x86_64-pc-windows-msvc-install_only.tar.gz"
        # probably can add macos support later
        printAndLog("Downloading Python")
        DownloadProgressPopup(
            link=link, downloadLocation=pyDir, title="Downloading Python"
        )

        # extract python
        self.extractTarGZ(pyDir)

        # give executable permissions to python
        makeExecutable(pythonPath())

    def downloadFFMpeg(self):
        ffmpegTempPath = os.path.join(currentDirectory(), "ffmpeg", "ffmpeg.temp")
        link = "https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/"
        match getPlatform():
            case "linux":
                link += "ffmpeg"
            case "win32":
                link += "ffmpeg.exe"

        printAndLog("Downloading FFMpeg")
        DownloadProgressPopup(
            link=link, downloadLocation=ffmpegTempPath, title="Downloading FFMpeg"
        )
        # give executable permissions to ffmpeg
        makeExecutable(ffmpegTempPath)
        move(ffmpegTempPath, ffmpegPath())

    def pipInstall(
        self, deps: list
    ):  # going to have to make this into a qt module pop up
        command = [pythonPath(), "-m", "pip", "install"] + deps
        printAndLog("Downloading Deps: " + str(command))
        DisplayCommandOutputPopup(command)

    def downloadPlatformIndependentDeps(self):
        platformIndependentdeps = [
            "testresources",
            "requests",
            "opencv-python-headless",
            "pypresence",
            "scenedetect",
            "numpy==1.26.4",
            "sympy",
        ]
        self.pipInstall(platformIndependentdeps)

    def downloadPyTorchCUDADeps(self):
        """
        Installs:
        Default deps
        Pytorch CUDA deps
        """
        self.downloadPlatformIndependentDeps()
        torchCUDALinuxDeps = [
            "spandrel",
            "https://download.pytorch.org/whl/nightly/pytorch_triton-3.0.0%2B45fff310c8-cp311-cp311-linux_x86_64.whl",
            "https://download.pytorch.org/whl/nightly/cu121/torch-2.5.0.dev20240620%2Bcu121-cp311-cp311-linux_x86_64.whl",
            "https://download.pytorch.org/whl/nightly/cu121/torchvision-0.20.0.dev20240620%2Bcu121-cp311-cp311-linux_x86_64.whl",
            "https://download.pytorch.org/whl/nightly/cu121/torch_tensorrt-2.5.0.dev20240620%2Bcu121-cp311-cp311-linux_x86_64.whl",
        ]
        torchCUDAWindowsDeps = [
            "spandrel",
            "https://download.pytorch.org/whl/nightly/cu121/torch-2.5.0.dev20240620%2Bcu121-cp311-cp311-win_amd64.whl",
            "https://download.pytorch.org/whl/nightly/cu121/torchvision-0.20.0.dev20240620%2Bcu121-cp311-cp311-win_amd64.whl"
            "https://download.pytorch.org/whl/nightly/cu121/torch_tensorrt-2.5.0.dev20240620%2Bcu121-cp311-cp311-win_amd64.whl",
        ]
        if getPlatform() == "win32":
            self.pipInstall(torchCUDAWindowsDeps)
        if getPlatform() == "linux":
            self.pipInstall(torchCUDALinuxDeps)

    def downloadNCNNDeps(self):
        """
        Installs:
        Default deps
        NCNN deps
        """
        self.downloadPlatformIndependentDeps()
        ncnnLinuxDeps = [
            "https://github.com/TNTwise/Universal-NCNN-upscaler-python/releases/download/2024-07-05/upscale_ncnn_py-1.2.0-cp311-none-manylinux1_x86_64.whl",
            "https://github.com/TNTwise/rife-ncnn-vulkan-python-test/releases/download/Revert_ncnn/rife_ncnn_vulkan_python-1.2.1-cp311-cp311-linux_x86_64.whl",
        ]
        ncnnWindowsDeps = [
            "https://github.com/TNTwise/Universal-NCNN-upscaler-python/releases/download/2024-07-05/upscale_ncnn_py-1.2.0-cp311-none-win_amd64.whl",
            "https://github.com/TNTwise/rife-ncnn-vulkan-python-test/releases/download/proc_bytes/rife_ncnn_vulkan_python-1.2.1-cp311-cp311-win_amd64.whl",
        ]
        if getPlatform() == "win32":
            self.pipInstall(ncnnWindowsDeps)
        if getPlatform() == "linux":
            self.pipInstall(ncnnLinuxDeps)

    def downloadPyTorchROCmDeps(self):
        pass

    def downloadTensorRTDeps(self):
        """
        Installs:
        Default deps
        Pytorch CUDA deps
        TensorRT deps
        """
        self.downloadPyTorchCUDADeps()
        tensorRTDeps = [
            "tensorrt==10.0.1",
            "tensorrt_cu12==10.0.1",
            "tensorrt-cu12_libs==10.0.1",
            "tensorrt_cu12_bindings==10.0.1",
        ]
        self.pipInstall(tensorRTDeps)


if __name__ == "__main__":
    downloadDependencies = DownloadDependencies()
    downloadDependencies.downloadPython()
