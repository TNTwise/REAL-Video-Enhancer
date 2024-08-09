from .Util import (
    getPlatform,
    printAndLog,
    pythonPath,
    ffmpegPath,
    currentDirectory,
    createDirectory,
    makeExecutable,
    move,
    extractTarGZ,
)
from .QTcustom import DownloadProgressPopup, DisplayCommandOutputPopup
import os
import subprocess


class DownloadDependencies:
    """
    Downloads platform specific dependencies python and ffmpeg to their respective locations and creates the directories

    """

    def __init__(self):
        createDirectory(os.path.join(currentDirectory(), "python"))
        createDirectory(os.path.join(currentDirectory(), "bin"))

    def get_total_dependencies(self, packages):
        total_dependencies = 0

        for package in packages:
            try:
                # Run pip show command and capture the output
                result = subprocess.run(
                    [pythonPath(), "-m", "pip", "show", "-v", package],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                # Parse the output to get the dependencies
                output = result.stdout.split("\n")
                dependencies = [
                    line.split(": ")[1]
                    for line in output
                    if line.startswith("Requires: ")
                ]

                # If there are dependencies, add their count to the total
                if dependencies:
                    total_dependencies += len(dependencies[0].split(", "))

            except subprocess.CalledProcessError:
                print(f"Warning: Package '{package}' not found or error occurred.")

        return total_dependencies

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
        extractTarGZ(pyDir)

        # give executable permissions to python
        makeExecutable(pythonPath())

    def downloadFFMpeg(self):
        ffmpegTempPath = os.path.join(currentDirectory(), "bin", "ffmpeg.temp")
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
        command = [pythonPath(), "-m", "pip", "install", "--upgrade", "pip", "--no-cache-dir"] + deps
        # totalDeps = self.get_total_dependencies(deps)
        totalDeps = len(deps)
        printAndLog("Downloading Deps: " + str(command))
        printAndLog("Total Dependencies: " + str(totalDeps))
        DisplayCommandOutputPopup(
            command=command,
            title="Download Dependencies",
            progressBarLength=totalDeps,
        )

    def getPlatformIndependentDeps(self):
        platformIndependentdeps = [
            "testresources",
            "requests",
            "opencv-python-headless",
            "pypresence",
            "scenedetect",
            "numpy==1.26.4",
            "sympy",
        ]
        return platformIndependentdeps

    def getPyTorchCUDADeps(self):
        """
        Installs:
        Default deps
        Pytorch CUDA deps
        """

        torchCUDALinuxDeps = [
            "https://github.com/TNTwise/spandrel/releases/download/sudo_span/spandrel-0.3.4-py3-none-any.whl",
            "https://download.pytorch.org/whl/nightly/pytorch_triton-3.0.0%2B45fff310c8-cp311-cp311-linux_x86_64.whl",
            "https://download.pytorch.org/whl/nightly/cu121/torch-2.5.0.dev20240620%2Bcu121-cp311-cp311-linux_x86_64.whl",
            "https://download.pytorch.org/whl/nightly/cu121/torchvision-0.20.0.dev20240620%2Bcu121-cp311-cp311-linux_x86_64.whl",
        ]
        torchCUDAWindowsDeps = [
            "https://github.com/TNTwise/spandrel/releases/download/sudo_span/spandrel-0.3.4-py3-none-any.whl",
            "https://download.pytorch.org/whl/nightly/cu121/torch-2.5.0.dev20240620%2Bcu121-cp311-cp311-win_amd64.whl",
            "https://download.pytorch.org/whl/nightly/cu121/torchvision-0.20.0.dev20240620%2Bcu121-cp311-cp311-win_amd64.whl",
        ]
        match getPlatform():
            case "win32":
                torchCUDAWindowsDeps += self.getPlatformIndependentDeps()
            case "linux":
                torchCUDALinuxDeps += self.getPlatformIndependentDeps()
        return torchCUDALinuxDeps

    def downloadPyTorchCUDADeps(self):
        self.pipInstall(self.getPyTorchCUDADeps() + self.getPlatformIndependentDeps())

    def downloadNCNNDeps(self):
        """
        Installs:
        Default deps
        NCNN deps
        """
        ncnnLinuxDeps = [
            "https://github.com/TNTwise/Universal-NCNN-upscaler-python/releases/download/2024-07-05/upscale_ncnn_py-1.2.0-cp311-none-manylinux1_x86_64.whl",
            "https://github.com/TNTwise/rife-ncnn-vulkan-python-test/releases/download/proc_bytes/rife_ncnn_vulkan_python-1.2.1-cp311-cp311-linux_x86_64.whl",
        ]
        ncnnWindowsDeps = [
            "https://github.com/TNTwise/Universal-NCNN-upscaler-python/releases/download/2024-07-05/upscale_ncnn_py-1.2.0-cp311-none-win_amd64.whl",
            "https://github.com/TNTwise/rife-ncnn-vulkan-python-test/releases/download/proc_bytes/rife_ncnn_vulkan_python-1.2.1-cp311-cp311-win_amd64.whl",
        ]
        match getPlatform():
            case "win32":
                self.pipInstall(ncnnWindowsDeps + self.getPlatformIndependentDeps())
            case "linux":
                self.pipInstall(ncnnLinuxDeps + self.getPlatformIndependentDeps())

    def downloadPyTorchROCmDeps(self):
        rocmLinuxDeps = [
            "https://download.pytorch.org/whl/pytorch_triton_rocm-2.3.1-cp311-cp311-linux_x86_64.whl",
            "https://download.pytorch.org/whl/rocm5.7/torch-2.3.1%2Brocm5.7-cp311-cp311-linux_x86_64.whl",
            "https://download.pytorch.org/whl/rocm5.7/torchvision-0.18.1%2Brocm5.7-cp311-cp311-linux_x86_64.whl",
        ]
        if getPlatform() == "linux":
            self.pipInstall(rocmLinuxDeps + self.getPlatformIndependentDeps())

    def getTensorRTDeps(self):
        """
        Installs:
        Default deps
        Pytorch CUDA deps
        TensorRT deps
        """
        tensorRTDeps = [
            "tensorrt==10.0.1",
            "tensorrt_cu12==10.0.1",
            "tensorrt-cu12_libs==10.0.1",
            "tensorrt_cu12_bindings==10.0.1",
        ]
        match getPlatform():
            case "linux":
                tensorRTDeps += (
                    "https://download.pytorch.org/whl/nightly/cu121/torch_tensorrt-2.5.0.dev20240620%2Bcu121-cp311-cp311-linux_x86_64.whl",
                )
            case "win32":
                tensorRTDeps += (
                    "https://download.pytorch.org/whl/nightly/cu121/torch_tensorrt-2.5.0.dev20240620%2Bcu121-cp311-cp311-win_amd64.whl",
                )
        return tensorRTDeps

    def downloadTensorRTDeps(self):
        self.pipInstall(
            self.getPyTorchCUDADeps() + self.getTensorRTDeps() + self.getTensorRTDeps()
        )


if __name__ == "__main__":
    downloadDependencies = DownloadDependencies()
    downloadDependencies.downloadPython()
