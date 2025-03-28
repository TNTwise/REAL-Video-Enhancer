from .constants import (
    PLATFORM,
    PYTHON_PATH,
    FFMPEG_PATH,
    BACKEND_PATH,
    TEMP_DOWNLOAD_PATH,
    CWD,
)
from .Util import (
    log,
    createDirectory,
    makeExecutable,
    move,
    extractTarGZ,
    downloadFile,
    removeFolder,
)
from .ui.QTcustom import (
    DownloadProgressPopup,
    DisplayCommandOutputPopup,
    RegularQTPopup,
)
import os
from platform import machine
import subprocess


def run_executable(exe_path):
    try:
        # Run the executable and wait for it to complete
        result = subprocess.run(exe_path, check=True, capture_output=True, text=True)

        # Print the output of the executable
        print("STDOUT:", result.stdout)

        # Print any error messages
        print("STDERR:", result.stderr)

        # Print the exit code
        print("Exit Code:", result.returncode)

    except subprocess.CalledProcessError as e:
        print("An error occurred while running the executable.")
        print("Exit Code:", e.returncode)
        print("Output:", e.output)
        print("Error:", e.stderr)
        return False
    except FileNotFoundError:
        print("The specified executable was not found.")
        return False
    except Exception as e:
        print("An unexpected error occurred:", str(e))
        return False
    return True


class DownloadDependencies:
    """
    Downloads platform specific dependencies python and ffmpeg to their respective locations and creates the directories

    """

    def __init__(self):
        if PLATFORM == "win32":
            self.__torchVersion = ""
        elif PLATFORM == "linux":
            self.__torchVersion = ""

        createDirectory(os.path.join(CWD, "python"))
        createDirectory(os.path.join(CWD, "bin"))

    def downloadBackend(self, tag):
        """
        Downloads the backend based on the tag of release.
        The tag of release is equal to the tag of the version.
        *NOTE
        tag is unused for now, as still in active development. just downloads the latest backend.
        """

        if not os.path.exists(BACKEND_PATH):
            print(str(BACKEND_PATH) + " Does not exist!")
            backend_url = "https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/backend-v2.2.0.tar.gz"
            main_zip = os.path.join(CWD, "backend.tar.gz")

            log("Downloading backend")
            downloadFile(link=backend_url, downloadLocation=main_zip)
            log("Extracting backend")
            extractTarGZ(main_zip)

    def downloadVCREDLIST(self):
        vcTempPath = os.path.join(CWD, "bin", "VC_redist.x64.exe")
        link = "https://aka.ms/vs/17/release/vc_redist.x64.exe"

        log("Downloading VC_redlist.x64.exe\nClick yes after download is complete.")
        DownloadProgressPopup(
            link=link,
            downloadLocation=vcTempPath,
            title="Downloading VC_redlist.x64.exe\nClick yes after download is complete.",
        )
        # give executable permissions to ffmpeg
        makeExecutable(vcTempPath)
        if not run_executable(
            [vcTempPath, "/install", "/quiet", "/norestart"]
        ):  # keep trying until user says yes
            RegularQTPopup(
                "Please click yes to allow VCRedlist to install!\nThe installer will now close."
            )

    def downloadPython(self):
        link = "https://github.com/indygreg/python-build-standalone/releases/download/20250317/cpython-3.12.9+20250317-"
        pyDir = os.path.join(
            CWD,
            "python",
            "python.tar.gz",
        )
        match PLATFORM:
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
        log("Downloading Python")
        DownloadProgressPopup(
            link=link, downloadLocation=pyDir, title="Downloading Python"
        )

        # extract python
        extractTarGZ(pyDir)

        # give executable permissions to python
        makeExecutable(PYTHON_PATH)

    def downloadFFMpeg(self):
        createDirectory(TEMP_DOWNLOAD_PATH)
        ffmpegTempPath = os.path.join(TEMP_DOWNLOAD_PATH, "ffmpeg")
        link = "https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/"
        match PLATFORM:
            case "linux":
                link += "ffmpeg"
            case "win32":
                link += "ffmpeg.exe"
            case "darwin":
                link += "ffmpeg-macos-bin"
        log("Downloading FFMpeg")
        DownloadProgressPopup(
            link=link, downloadLocation=ffmpegTempPath, title="Downloading FFMpeg"
        )
        # give executable permissions to ffmpeg
        makeExecutable(ffmpegTempPath)
        move(ffmpegTempPath, FFMPEG_PATH)
        removeFolder(TEMP_DOWNLOAD_PATH)

    def pip(
        self,
        deps: list,
        install: bool = True,
    ):  # going to have to make this into a qt module pop up
        createDirectory(TEMP_DOWNLOAD_PATH)
        command = [
            PYTHON_PATH,
            "-m",
            "pip",
            "install" if install else "uninstall",
        ]
        origTemp = os.environ.get("TMPDIR")
        os.environ["TMPDIR"] = TEMP_DOWNLOAD_PATH
        if install:
            command += [
                "--no-warn-script-location",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu126",
                "--extra-index-url",
                "https://pypi.nvidia.com",
            ]
        else:
            command += ["-y"]
        command += deps
        # totalDeps = self.get_total_dependencies(deps)
        totalDeps = len(deps)
        log("Downloading Deps: " + str(command))
        log("Total Dependencies: " + str(totalDeps))

        DisplayCommandOutputPopup(
            command=command,
            title="Download Dependencies",
            progressBarLength=totalDeps,
        )
        command = [
            PYTHON_PATH,
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
        if origTemp:
            os.environ["TMPDIR"] = str(origTemp)
        removeFolder(TEMP_DOWNLOAD_PATH)

    def getPlatformIndependentDeps(self):
        platformIndependentdeps = [
            "testresources",
            "requests",
            "opencv-python-headless",
            "pypresence",
            "scenedetect",
            "numpy==2.2.2",
            "sympy==1.13.1",
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
        torchCUDADeps = [
            f"torch==2.6.0",
            f"torchvision==0.21.0",
            "safetensors",
            "einops",
            "cupy-cuda12x==13.3.0",
        ]
        return torchCUDADeps

    def getTensorRTDeps(self):
        """
        Installs:
        Default deps
        Pytorch CUDA deps
        TensorRT deps
        """
        deps = [
            "tensorrt==10.9.0.34",
            "tensorrt_cu12==10.9.0.34",
            "tensorrt-cu12_libs==10.9.0.34",
            "tensorrt_cu12_bindings==10.9.0.34",
            "--no-deps",
            "torch_tensorrt==2.6.0",
        ]

        return deps

    def downloadPyTorchCUDADeps(self, install: bool = True):
        if install:
            self.pip(self.getPlatformIndependentDeps())
        self.pip(self.getPyTorchCUDADeps(), install)

    def downloadTensorRTDeps(self, install: bool = True):
        if install:
            self.pip(self.getPlatformIndependentDeps())
        self.pip(
            self.getPyTorchCUDADeps(),
            install,
        )
        self.pip(
            self.getTensorRTDeps(),  # Has to be in this order, because i skip dependency check for torchvision
            install,
        )

    def downloadDirectMLDeps(self, install: bool = True):
        directMLDeps = [
            "onnxruntime-directml",
            "onnx",
            "onnxconverter-common",
        ] + self.getPlatformIndependentDeps()
        self.pip(directMLDeps, install)

    def downloadNCNNDeps(self, install: bool = True):
        """
        Installs:
        Default deps
        NCNN deps
        """
        if install:
            self.pip(self.getPlatformIndependentDeps())
        ncnnDeps = [
            "rife-ncnn-vulkan-python-tntwise==1.4.5",
            "upscale_ncnn_py==1.2.0",
            "ncnn==1.0.20240820",
            "numpy==2.2.2",
            "opencv-python-headless",
            "mpmath",
            "sympy==1.13.1",
        ]
        self.pip(ncnnDeps, install)

    def downloadPyTorchROCmDeps(self, install: bool = True):
        if install:
            self.pip(self.getPlatformIndependentDeps())

        rocmLinuxDeps = [
            "https://download.pytorch.org/whl/pytorch_triton_rocm-2.3.1-cp312-cp312-linux_x86_64.whl",
            "einops",
            "safetensors",
            "https://download.pytorch.org/whl/rocm5.7/torch-2.3.1%2Brocm5.7-cp312-cp312-linux_x86_64.whl",
            "https://download.pytorch.org/whl/rocm5.7/torchvision-0.18.1%2Brocm5.7-cp312-cp312-linux_x86_64.whl",
        ]
        if PLATFORM == "linux":
            self.pip(rocmLinuxDeps, install)


if __name__ == "__main__":
    downloadDependencies = DownloadDependencies()
    downloadDependencies.downloadPython()
