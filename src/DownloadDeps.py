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
    downloadTempDirectory,
    downloadFile,
    backendDirectory,
)
from .ui.QTcustom import DownloadProgressPopup, DisplayCommandOutputPopup
import os
from platform import machine


class DownloadDependencies:
    """
    Downloads platform specific dependencies python and ffmpeg to their respective locations and creates the directories

    """

    def __init__(self):
        createDirectory(os.path.join(currentDirectory(), "python"))
        createDirectory(os.path.join(currentDirectory(), "bin"))
        createDirectory(os.path.join(currentDirectory(), "pip_cache"))

    def downloadBackend(self, tag):
        """
        Downloads the backend based on the tag of release.
        The tag of release is equal to the tag of the version.
        *NOTE
        tag is unused for now, as still in active development. just downloads the latest backend.
        """

        if not os.path.exists(backendDirectory()):
            print(str(backendDirectory()) + " Does not exist!")
            backend_url = "https://github.com/TNTwise/real-video-enhancer-models/releases/download/flatpak-backends/backend-V2-stable.tar.gz"
            main_zip = os.path.join(currentDirectory(), "backend.tar.gz")
            main_folder = os.path.join(currentDirectory(), "repo")
            orig_backend_folder = os.path.join(
                main_folder, "REAL-Video-Enhancer-2.0", "backend"
            )
            moved_backed_folder = os.path.join(currentDirectory(), "backend")

            printAndLog("Downloading backend")
            # urllib.request.urlretrieve(backend_url, main_zip)
            downloadFile(link=backend_url, downloadLocation=main_zip)
            printAndLog("Extracting backend")
            extractTarGZ(main_zip)
            # printAndLog("Extracting backend")
            # shutil.unpack_archive(main_zip, main_folder)
            # printAndLog("Moving Backend")
            # move(orig_backend_folder, moved_backed_folder)
            # printAndLog("Cleaning up")

    def downloadPython(self):
        link = "https://github.com/indygreg/python-build-standalone/releases/download/20240814/cpython-3.11.9+20240814-"
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
                if machine() == "arm64":
                    link += "aarch64-apple-darwin-install_only.tar.gz"
                else:
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
        ffmpegTempPath = os.path.join(downloadTempDirectory(), "ffmpeg")
        link = "https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/"
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
            "--cache-dir=" + os.path.join(currentDirectory(), "pip_cache"),
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
            # "https://download.pytorch.org/whl/nightly/pytorch_triton-3.0.0%2B45fff310c8-cp311-cp311-linux_x86_64.whl",
            "torch==2.4.0",
            "torchvision==0.19.0",
        ]"""
        # Nigthly test
        torchCUDALinuxDeps = [
            #"https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/spandrel-0.3.4-py3-none-any.whl",
            "https://download.pytorch.org/whl/nightly/pytorch_triton-3.0.0%2Bdedb7bdf33-cp311-cp311-linux_x86_64.whl",
            "https://download.pytorch.org/whl/nightly/cu124_pypi_pkg/torch-2.5.0.dev20240826%2Bcu124-cp311-cp311-linux_x86_64.whl",
            "https://download.pytorch.org/whl/nightly/cu124/torchvision-0.20.0.dev20240826%2Bcu124-cp311-cp311-linux_x86_64.whl",
            "https://download.pytorch.org/whl/nightly/cu124_pypi_pkg/torch_no_python-2.5.0.dev20240826%2Bcu124-py3-none-any.whl",
            "safetensors",
            "einops",
            "cupy-cuda12x==13.3.0",
        ]
        torchCUDAWindowsDeps = [
            #"https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/spandrel-0.3.4-py3-none-any.whl",
            # "--pre",
            "https://download.pytorch.org/whl/nightly/cu124/torch-2.5.0.dev20240826%2Bcu124-cp311-cp311-win_amd64.whl",
            # "--pre",
            "https://download.pytorch.org/whl/nightly/cu124/torchvision-0.20.0.dev20240826%2Bcu124-cp311-cp311-win_amd64.whl",
            # "torch==2.4.0",
            # "torchvision==0.19.0",
            "safetensors",
            "einops",
            "cupy-cuda12x==13.3.0",
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
                # tensorRTDeps += [  # "--no-deps",
                #    "torch_tensorrt==2.4.0"]

                # nightly
                tensorRTDeps += [
                    "https://download.pytorch.org/whl/nightly/cu124/torch_tensorrt-2.5.0.dev20240826%2Bcu124-cp311-cp311-linux_x86_64.whl"
                ]
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
                    "https://download.pytorch.org/whl/nightly/cu124/torch_tensorrt-2.5.0.dev20240826%2Bcu124-cp311-cp311-win_amd64.whl",
                )
        return tensorRTDeps

    def downloadPyTorchCUDADeps(self):
        self.pipInstall(self.getPyTorchCUDADeps())

    def downloadTensorRTDeps(self):
        self.pipInstall(
            self.getPyTorchCUDADeps()
            + self.getTensorRTDeps()  # Has to be in this order, because i skip dependency check for torchvision
        )

    def downloadDirectMLDeps(self):
        directMLDeps = [
            "onnxruntime-directml",
            "onnx",
            "onnxconverter-common",
        ] + self.getPlatformIndependentDeps()
        self.pipInstall(directMLDeps)

    def downloadNCNNDeps(self):
        """
        Installs:
        Default deps
        NCNN deps
        """
        ncnnDeps = [
            "rife-ncnn-vulkan-python-tntwise==1.4.2",
            "upscale_ncnn_py==1.2.0",
            "ncnn==1.0.20240820",
            "numpy==1.26.4",
            "opencv-python-headless",
        ] + self.getPlatformIndependentDeps()
        self.pipInstall(ncnnDeps)
        self.pipInstall(['numpy==1.26.4','sympy'])

    def downloadPyTorchROCmDeps(self):
        rocmLinuxDeps = [
            "https://download.pytorch.org/whl/pytorch_triton_rocm-2.3.1-cp311-cp311-linux_x86_64.whl",
            "https://download.pytorch.org/whl/rocm5.7/torch-2.3.1%2Brocm5.7-cp311-cp311-linux_x86_64.whl",
            "https://download.pytorch.org/whl/rocm5.7/torchvision-0.18.1%2Brocm5.7-cp311-cp311-linux_x86_64.whl",
        ]
        if getPlatform() == "linux":
            self.pipInstall(rocmLinuxDeps + self.getPlatformIndependentDeps())


if __name__ == "__main__":
    downloadDependencies = DownloadDependencies()
    downloadDependencies.downloadPython()