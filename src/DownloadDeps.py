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
from .ui.QTcustom import DownloadProgressPopup, DisplayCommandOutputPopup
import os
import subprocess
import shutil
import urllib.request


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

    def downloadBackend(self, tag):
        """
        Downloads the backend based on the tag of release.
        The tag of release is equal to the tag of the version.
        *NOTE
        tag is unused for now, as still in active development. just downloads the latest backend.
        """
        if not os.path.exists(os.path.join(currentDirectory(), "backend")):
            backend_url = "https://github.com/tntwise/REAL-Video-Enhancer/archive/refs/heads/2.0.zip"
            main_zip = os.path.join(currentDirectory(), "repo.zip")
            main_folder = os.path.join(currentDirectory(), "repo")
            orig_backend_folder = os.path.join(
                main_folder, "REAL-Video-Enhancer-2.0", "backend"
            )
            moved_backed_folder = os.path.join(currentDirectory(), "backend")

            printAndLog("Downloading backend")
            urllib.request.urlretrieve(backend_url, main_zip)
            # DownloadProgressPopup(link=backend_url, downloadLocation=backend_zip,title="Downloading Backend")

            printAndLog("Extracting backend")
            shutil.unpack_archive(main_zip, main_folder)
            printAndLog("Moving Backend")
            move(orig_backend_folder, moved_backed_folder)
            printAndLog("Cleaning up")
            os.remove(main_zip)

    def downloadPython(self):
        link = "https://github.com/indygreg/python-build-standalone/releases/download/20240814/cpython-3.12.5+20240814-"
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
            case "darwin":
                link += "x86_64-apple-darwin-install_only.tar.gz"
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
            case "darwin":
                link += "ffmpeg-macos-bin"
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
        command = [
            pythonPath(),
            "-m",
            "pip",
            "install",
            "-U",
            "--no-warn-script-location",
        ] + deps
        # totalDeps = self.get_total_dependencies(deps)
        totalDeps = len(deps)
        printAndLog("Downloading Deps: " + str(command))
        printAndLog("Total Dependencies: " + str(totalDeps))
        DisplayCommandOutputPopup(
            command=command,
            title="Download Dependencies",
            progressBarLength=totalDeps,
        )
        command = [
            pythonPath(),
            "-m",
            "pip",
            "cache",
            "purge",
        ]
        DisplayCommandOutputPopup(
            command=command,
            title="Purging Cache",
            progressBarLength=1,
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
            "tqdm",
            "typing_extensions",
            "packaging",
            "mpmath",
            "pillow",
        ]
        return platformIndependentdeps

    def getPyTorchCUDADeps(self):
        """
        Installs:
        Default deps
        Pytorch CUDA deps
        """
        # default
        """torchCUDALinuxDeps = [
            "https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/spandrel-0.3.4-py3-none-any.whl",
            # "https://download.pytorch.org/whl/nightly/pytorch_triton-3.0.0%2B45fff310c8-cp312-cp312-linux_x86_64.whl",
            "torch==2.4.0",
            "torchvision==0.19.0",
        ]"""
        # Nigthly test
        torchCUDALinuxDeps = [
            "https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/spandrel-0.3.4-py3-none-any.whl",
            "https://download.pytorch.org/whl/nightly/pytorch_triton-3.0.0%2Bdedb7bdf33-cp312-cp312-linux_x86_64.whl",
            "https://download.pytorch.org/whl/nightly/cu124_pypi_pkg/torch-2.5.0.dev20240826%2Bcu124-cp312-cp312-linux_x86_64.whl",
            "https://download.pytorch.org/whl/nightly/cu124/torchvision-0.20.0.dev20240826%2Bcu124-cp312-cp312-linux_x86_64.whl",
            "https://download.pytorch.org/whl/nightly/cu124_pypi_pkg/torch_no_python-2.5.0.dev20240826%2Bcu124-py3-none-any.whl"
        ]
        torchCUDAWindowsDeps = [
            "https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/spandrel-0.3.4-py3-none-any.whl",
            # "--pre",
            "https://download.pytorch.org/whl/nightly/cu124/torch-2.5.0.dev20240826%2Bcu124-cp312-cp312-win_amd64.whl",
            # "--pre",
            "https://download.pytorch.org/whl/nightly/cu124/torchvision-0.20.0.dev20240826%2Bcu124-cp312-cp312-win_amd64.whl",
            # "torch==2.4.0",
            # "torchvision==0.19.0",
            # "safetensors",
            # "einops",
        ]
        match getPlatform():
            case "win32":
                return (
                    self.getPlatformIndependentDeps() + torchCUDAWindowsDeps
                )  # flipped order for skipping check on deps with torchvision
            case "linux":
                return torchCUDALinuxDeps + self.getPlatformIndependentDeps()

    

    

    def getTensorRTDeps(self):
        """
        Installs:
        Default deps
        Pytorch CUDA deps
        TensorRT deps
        """

        match getPlatform():
            case "linux":
                tensorRTDeps = [
                    "tensorrt==10.3.0",
                    "tensorrt_cu12==10.3.0",
                    "tensorrt-cu12_libs==10.3.0",
                    "tensorrt_cu12_bindings==10.3.0",
                ]
                # default
                #tensorRTDeps += [  # "--no-deps",
                #    "torch_tensorrt==2.4.0"]
                
                # nightly
                tensorRTDeps += ["https://download.pytorch.org/whl/nightly/cu124/torch_tensorrt-2.5.0.dev20240826%2Bcu124-cp312-cp312-linux_x86_64.whl"]
            case "win32":
                tensorRTDeps = [
                    "tensorrt==10.3.0",
                    "tensorrt_cu12==10.3.0",
                    "tensorrt-cu12_libs==10.3.0",
                    "tensorrt_cu12_bindings==10.3.0",
                ]
                tensorRTDeps += (
                    # "--no-deps",
                    # "torch_tensorrt==2.4.0",
                    "https://download.pytorch.org/whl/nightly/cu124/torch_tensorrt-2.5.0.dev20240826%2Bcu124-cp312-cp312-win_amd64.whl",
                )
        return tensorRTDeps

    def downloadPyTorchCUDADeps(self):
        self.pipInstall(self.getPyTorchCUDADeps())
    
    def downloadTensorRTDeps(self):
        self.pipInstall(
            self.getPyTorchCUDADeps()
            + self.getTensorRTDeps()  # Has to be in this order, because i skip dependency check for torchvision
        )

    def downloadNCNNDeps(self):
        """
        Installs:
        Default deps
        NCNN deps
        """
        ncnnDeps = [
            "rife-ncnn-vulkan-python-tntwise==1.4.1",
            "upscale_ncnn_py==1.2.0",
        ]  + self.getPlatformIndependentDeps()
        self.pipInstall(ncnnDeps)


    def downloadPyTorchROCmDeps(self):
        rocmLinuxDeps = [
            "https://download.pytorch.org/whl/pytorch_triton_rocm-2.3.1-cp312-cp312-linux_x86_64.whl",
            "https://download.pytorch.org/whl/rocm5.7/torch-2.3.1%2Brocm5.7-cp312-cp312-linux_x86_64.whl",
            "https://download.pytorch.org/whl/rocm5.7/torchvision-0.18.1%2Brocm5.7-cp312-cp312-linux_x86_64.whl",
        ]
        if getPlatform() == "linux":
            self.pipInstall(rocmLinuxDeps + self.getPlatformIndependentDeps())

    


if __name__ == "__main__":
    downloadDependencies = DownloadDependencies()
    downloadDependencies.downloadPython()
