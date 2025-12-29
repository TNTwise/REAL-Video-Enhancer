import os
import sys
import platform
CPU_ARCH = "x86_64" if platform.machine() == "AMD64" else platform.machine()

__version__ = "2.1.5"
IS_FLATPAK = "FLATPAK_ID" in os.environ
HOME_PATH = os.path.expanduser("~")
PLATFORM = sys.platform  # win32, darwin, linux

USE_LOCAL_BACKEND = os.path.exists(
    os.path.join(os.getcwd(), "backend") # gets if the backend folder is in the current directory, if not switch the directories
)
if not USE_LOCAL_BACKEND:
    
    if PLATFORM == "win32":
        CWD = os.path.join(HOME_PATH, "AppData", "Local", "REAL-Video-Enhancer")
    if PLATFORM == "darwin":
        CWD = os.path.join(HOME_PATH, "Library", "REAL-Video-Enhancer")
    if PLATFORM == "linux":
        CWD = os.path.join(
            HOME_PATH, ".local", "share", "REAL-Video-Enhancer"
        )
else:
    CWD = os.getcwd()
    
if IS_FLATPAK:
        CWD = (
            os.path.join(
                HOME_PATH, ".var", "app", "io.github.tntwise.REAL-Video-Enhancer"
            )
        )

FFMPEG_LOG_FILE = os.path.join(CWD, "ffmpeg_log.txt")
MODELS_DIRECTORY = os.path.join(CWD, "models")
