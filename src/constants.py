import os
import sys
import requests
import platform
import subprocess
from PySide6.QtCore import QDir


def networkCheck(hostname="https://raw.githubusercontent.com") -> bool:
    """
    checks network availability against a url, default url: raw.githubusercontent.com
    """
    try:
        _ = requests.head(hostname, timeout=1)
        return True
    except Exception as e:
        pass
    return False




HAS_NETWORK_ON_STARTUP = networkCheck()

LOCKFILE = QDir.tempPath() + "/REAL-Video-Enhancer.lock"


PLATFORM = sys.platform.strip().lower()  # win32, darwin, linux
IS_STEAM = "SteamAppId" in os.environ
IS_FLATPAK = "FLATPAK_ID" in os.environ and not IS_STEAM

HOME_PATH = os.path.expanduser("~")

IS_COMPILED_OR_FROZEN = hasattr(sys, "frozen")
USE_LOCAL_BACKEND = os.path.exists(os.path.join(os.getcwd(), "backend"))

if not USE_LOCAL_BACKEND:
    if PLATFORM == "win32":
        CWD = os.path.join(HOME_PATH, "AppData", "Local", "REAL-Video-Enhancer")
    if PLATFORM == "darwin":
        CWD = os.path.join(HOME_PATH, "Library", "REAL-Video-Enhancer")
    if PLATFORM == "linux":
        CWD = os.path.join(HOME_PATH, ".local", "share", "REAL-Video-Enhancer")
else:
    CWD = os.getcwd()

if IS_FLATPAK:
    CWD = os.path.join(
        HOME_PATH, ".var", "app", "io.github.tntwise.REAL-Video-Enhancer"
    )


CPU_ARCH = "x86_64" if platform.machine() == "AMD64" else platform.machine()
if CPU_ARCH.lower() == "arm64" or CPU_ARCH.lower() == "aarch64":
    CPU_ARCH = "arm64"

def getCPUInfo() -> str:
    """
    Returns the CPU information of the system.
    """
    # return platform.processor() + " " + str(psutil.cpu_count(logical=False)) + " cores" + platform.
    try:
        if PLATFORM == "win32":
            try:
                # Run the 'wmic' command to get CPU information
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                # Split the result by lines and return the second line which contains the CPU name
                return result.stdout.split("\n")[2].strip()
            except Exception as e:
                return "X86_64 CPU" if CPU_ARCH == "x86_64" else "ARM64 CPU"
        else:
            import cpuinfo
            return cpuinfo.get_cpu_info()["brand_raw"]
    except Exception as e:
        return "Unknown"
    
CPU_INFO = getCPUInfo()

if "apple" in CPU_INFO.lower():
    CPU_ARCH = "arm64"


EXE_NAME = "REAL-Video-Enhancer.exe" if PLATFORM == "win32" else "REAL-Video-Enhancer"
LIBS_NAME = "_internal" if PLATFORM == "win32" else "lib"
# dirs
MODELS_PATH = os.path.join(CWD, "models")
CUSTOM_MODELS_PATH = os.path.join(CWD, "custom_models")
PRESETS_PATH = os.path.join(CWD, "presets")
VIDEOS_PATH = (
    os.path.join(HOME_PATH, "Desktop")
    if PLATFORM == "darwin"
    else os.path.join(HOME_PATH, "Videos")
)
BACKEND_PATH = "/app/bin/backend" if IS_FLATPAK else os.path.join(CWD, "backend")
TEMP_DOWNLOAD_PATH = os.path.join(CWD, "temp")
# exes
FFMPEG_PATH = (
    os.path.join(CWD, "bin", "ffmpeg.exe")
    if PLATFORM == "win32"
    else os.path.join(CWD, "bin", "ffmpeg")
)
PYTHON_DIRECTORY = os.path.join(CWD, "python")

PYTHON_EXECUTABLE_PATH = (
    os.path.join(CWD, "python", "python", "python.exe")
    if PLATFORM == "win32"
    else os.path.join(CWD, "python", "python", "bin", "python3")
)
# PYTHON_VERSION = "3.13.2" if PLATFORM != "darwin" else "3.12.9" # sets python version of backend
PYTHON_VERSION = "3.12.9"

EXE_PATH = os.path.join(
    CWD,
    EXE_NAME,
)
LIBS_PATH = os.path.join(
    CWD,
    LIBS_NAME,
)

# is installed
IS_INSTALLED = os.path.isfile(FFMPEG_PATH) and os.path.isfile(PYTHON_EXECUTABLE_PATH)

IMAGE_SHARED_MEMORY_ID = "/image_preview" + str(os.getpid())
PAUSED_STATE_SHARED_MEMORY_ID = "/paused_state" + str(os.getpid())
INPUT_TEXT_FILE = os.path.join(CWD, f"INPUT{os.getpid()}.txt")
if (
    "--swap-flatpak-checks" in sys.argv
):  # swap check down here as to not interfere with directories
    IS_FLATPAK = not IS_FLATPAK
