from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from PySide6.QtWidgets import QMessageBox

from .constants import (
    PLATFORM,
    PYTHON_DIRECTORY,
    PYTHON_EXECUTABLE_PATH,
    PYTHON_VERSION,
    FFMPEG_PATH,
    BACKEND_PATH,
    TEMP_DOWNLOAD_PATH,
    CWD,
    CPU_ARCH,
    USE_LOCAL_BACKEND,
    IS_STEAM
)
from .version import version, backend_dev_version
from .Util import (
    FileHandler,
    log,
    extractTarGZ,
    removeFolder,
    subprocess_popen_without_terminal
)
from .ui.QTcustom import (
    DownloadProgressPopup,
    DisplayCommandOutputPopup,
    RegularQTPopup,
    needs_network_else_exit,
)
import os
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

@dataclass
class Dependency(ABC):
    updatable: bool
    download_path:str
    installed_path:str

    def __init__(self):
        FileHandler.createDirectory(os.path.dirname(self.download_path))

    @abstractmethod
    def get_download_link(self) -> str: ...
        
    @abstractmethod
    def download(self) -> None: ...

    def get_if_update_available(self) -> bool: ...
    def update_if_updates_available(self) -> None: ...

class Backend(Dependency):
    updatable: bool = True
    is_update_available: bool = False
    download_path = os.path.join(CWD, "backend.tar.gz")
    installed_path = BACKEND_PATH


    def get_download_link(self) -> str:
        backend_url = f"https://github.com/TNTwise/REAL-Video-Enhancer/releases/download/RVE-{version}/backend-v{version}.tar.gz"
        return backend_url
    
    def download(self):
        if USE_LOCAL_BACKEND:
            return
        needs_network_else_exit()
        download_link = self.get_download_link()
        DownloadProgressPopup(link=download_link, downloadLocation=self.download_path, title="Downloading Backend")
        extractTarGZ(self.download_path)
    
    def get_if_update_available(self) -> bool:
        try:
            process = subprocess_popen_without_terminal(
                [PYTHON_EXECUTABLE_PATH, os.path.join(BACKEND_PATH, "rve-backend.py"), "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
                )
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, process.args, stdout, stderr)
            output = stdout.strip() # this extracts the version number from the output
            log(f"\nBackend Version: {output}\n")
            update_available = not output == backend_dev_version
            self.is_update_available = update_available
            return update_available
        except subprocess.CalledProcessError as e: # if the backend is not found
            log("Backend not found, downloading..." + str(e))
            self.download()
            self.is_update_available = False
            return False
    
    def update_if_updates_available(self) -> None:
        if self.is_update_available:
            needs_network_else_exit()
            FileHandler.removeFolder(BACKEND_PATH) # remove the old backend directory
            self.download()


class Python(Dependency):
    download_path = os.path.join(CWD, "python", "python.tar.gz")
    installed_path = PYTHON_DIRECTORY
    is_update_available: bool

    def get_download_link(self) -> str:
        link = f"https://github.com/TNTwise/REAL-Video-Enhancer-models/releases/download/models/cpython-{PYTHON_VERSION}+20250317-"
       
        match PLATFORM:
            case "linux":
                link += "x86_64-unknown-linux-gnu-install_only.tar.gz" if CPU_ARCH == "x86_64" else "aarch64-unknown-linux-gnu-install_only.tar.gz"
            case "win32":
                link += "x86_64-pc-windows-msvc-install_only.tar.gz"
            case "darwin":
                link += "x86_64-apple-darwin-install_only.tar.gz" if CPU_ARCH == "x86_64" else "aarch64-apple-darwin-install_only.tar.gz"

        return link

    def download(self):
        needs_network_else_exit()
        download_link = self.get_download_link()
        FileHandler.createDirectory(os.path.dirname(self.download_path))
        DownloadProgressPopup(link = download_link, downloadLocation=self.download_path, title = f"Downloading Python {PYTHON_VERSION}")
        extractTarGZ(self.download_path)
    
    def get_version(self):
        return subprocess.run([PYTHON_EXECUTABLE_PATH, "--version"], check=True, capture_output=True, text=True).stdout.strip().split(" ")[1] # this extracts the version number from the output
    
    def get_if_update_available(self) -> bool:
        try:
            output = self.get_version()
        except subprocess.CalledProcessError: # if python is not found
            self.download()
            return False
        
        is_update = not output == PYTHON_VERSION

        if is_update:
            reply = QMessageBox.question(
                None,
                "Update Python?",
                "The installed version of Python is older than the current version of RVE. Update?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,  # type: ignore
            )
            if reply == QMessageBox.Yes:  # type: ignore
                self.is_update_available = True
                print("Updating Python")
            else:
                is_update = False
        self.is_update_available = is_update
        return self.is_update_available

    def update_if_updates_available(self) -> None:

        if self.is_update_available:
            removeFolder(PYTHON_DIRECTORY)
            self.download()

class FFMpeg(Dependency):
    download_path = os.path.join(CWD, "ffmpeg")
    installed_path = FFMPEG_PATH

    def get_download_link(self) -> str:
        link = "https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/"
        match PLATFORM:
            case "linux":
                link += "ffmpeg" if CPU_ARCH == "x86_64" else "ffmpeg-linux-arm64"
            case "win32":
                link += "ffmpeg.exe" if CPU_ARCH == "x86_64" else "ffmpeg-windows-arm64.exe"
            case "darwin":
                link += "ffmpeg-macos-bin" if CPU_ARCH == "x86_64" else "ffmpeg-macos-arm"
        return link

    def download(self):
        
        needs_network_else_exit()

        download_link = self.get_download_link()
        DownloadProgressPopup(link=download_link, downloadLocation=self.download_path, title="Downloading FFMpeg")
        FileHandler.createDirectory(os.path.dirname(self.installed_path))
        FileHandler.moveFile(self.download_path, self.installed_path)
        FileHandler.makeExecutable(self.installed_path)

class VCRedList(Dependency):
    updatable = False
    download_path = os.path.join(CWD, "bin", "VC_redist.x64.exe")
    installed_path = download_path

    def get_download_link(self) -> str:
        return "https://aka.ms/vs/17/release/vc_redist.x64.exe"
    
    def download(self):
        if PLATFORM == 'win32':
            needs_network_else_exit()

            download_link = self.get_download_link()
            DownloadProgressPopup(link=download_link, downloadLocation=self.download_path, title="Downloading VCRedist")
            
            # Use ShellExecute to properly handle admin elevation
            import ctypes
            try:
                result = ctypes.windll.shell32.ShellExecuteW(
                    None,                          # hwnd
                    "runas",                       # operation (runas = run as admin)
                    self.download_path,            # file
                    "/install /norestart /quiet",         # parameters
                    None,                          # directory
                    1                              # show command (1 = normal window)
                )
                if result <= 32:  # Error codes are <= 32
                    RegularQTPopup(
                        "Failed to launch VCRedist installer. Please run it manually."
                    )
            except Exception as e:
                RegularQTPopup(
                    f"Error launching VCRedist installer: {str(e)}\nThe installer will now close."
                )




class DownloadDependencies:
    """
    Downloads platform specific dependencies python and ffmpeg to their respective locations and creates the directories

    """
    def __init__(self, use_torch_nightly:bool = False):
        self.use_torch_nightly = use_torch_nightly
    def download_all_deps(self):
        for dep in Dependency.__subclasses__():
            d = dep()
            d.download()

    def pip(
        self,
        deps: list,
        install: bool = True,
    ):  # going to have to make this into a qt module pop up
        command = []
        
        command += [
            PYTHON_EXECUTABLE_PATH,
            "-m",
            "pip",
            "install" if install else "uninstall",
        ]
        #origTemp = os.environ.get("TMPDIR")
        #os.environ["TMPDIR"] = TEMP_DOWNLOAD_PATH
        if install:
            command += [
                "--no-warn-script-location",
                "--isolated",
                "--extra-index-url",
                "https://download.pytorch.org/whl/test/", 
                "--extra-index-url",
                "https://download.pytorch.org/whl/", # search this first, needs to be last in the list 
            ]
        else:
            command += ["-y"]
        command += deps
        # totalDeps = self.get_total_dependencies(deps)
        totalDeps = len(deps)
        log("Downloading Deps: " + str(command))
        log("Total Dependencies: " + str(totalDeps))

        d = DisplayCommandOutputPopup(
            command=command,
            title="Download Dependencies",
            progressBarLength=totalDeps,
        )

        return_code = d.get_return_code()

        command = [
            PYTHON_EXECUTABLE_PATH,
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
        #if origTemp:
        #    os.environ["TMPDIR"] = str(origTemp)
        return return_code

    def getPlatformIndependentDeps(self):
        platformIndependentdeps = [
            "testresources==2.0.1",
            "requests==2.32.3",
            "opencv-python-headless==4.11.0.86",
            "pypresence==4.3.0",
            "scenedetect==0.6.5.2",
            "numpy==2.2.2",
            "sympy",
            "tqdm==4.67.1",
            "typing_extensions==4.12.2",
            "packaging==24.2",
            "mpmath==1.3.0",
            "pillow==11.1.0",
        ]
        return platformIndependentdeps
    
    def downloadPythonDeps(self, backend, torch_version: Optional[str] = "2.7.0", torchvision_version: Optional[str] = "0.22.0", torch_backend: Optional[str] = "cu126", install: bool = True):
        deps = []
        log("Downloading Python Deps for " + backend)
        log("Torch Version: " + torch_version)
        log("Torch Backend: " + torch_backend)
        log("Torchvision Version: " + torchvision_version)
        

        if install: # dont uninstall platform independent deps
            deps = self.getPlatformIndependentDeps()
            
        return_codes = []
        match backend:
            case "ncnn":
                deps += [
                    "rife-ncnn-vulkan-python-tntwise==1.4.5",
                    "upscale_ncnn_py==1.2.0",
                    "ncnn==1.0.20240820",
                    "numpy==2.2.2",
                ]
                return_code = self.pip(deps, install)
                return_codes.append(return_code)
            case "torch" | "tensorrt":
                deps += [
                    f"torch=={torch_version}{torch_backend}",  #
                    "safetensors==0.5.3",
                    "einops==0.8.1",
                    
                ]
                deps += ["cupy-cuda12x==13.3.0"] if "cu" in backend else []
                return_code = self.pip(deps, install)
                return_codes.append(return_code)
                
                if install:
                    deps = [
                        "--no-deps",
                        f"torchvision=={torchvision_version}{torch_backend}",
                    ]
                    return_code = self.pip(deps, install)

                return_codes.append(return_code)

                if backend == "tensorrt":
                    trt_ver = "10.12.0.36"
                    deps = [
                        f"tensorrt=={trt_ver}",
                        f"tensorrt_cu12=={trt_ver}",
                        f"tensorrt-cu12_libs=={trt_ver}",
                        f"tensorrt_cu12_bindings=={trt_ver}",
                        
                    ]
                    if install:
                        
                        torch_version = torch_version[:-1] + "0" # remove the last character (2.7.1 -> 2.7.0), torch tensorrt doesnt release a new version for every new pytorch minor release
                        deps += ["--no-deps","dllist",f"torch-tensorrt=={torch_version}{torch_backend}"]

                    return_code = self.pip(deps, install)
                    return_codes.append(return_code)
        
        for return_code in return_codes:
            if return_code != 0:
                return return_code
        return 0