import os
import platform
import sys

CPU_ARCH = "x86_64" if platform.machine() == "AMD64" else platform.machine()
IS_FLATPAK = "FLATPAK_ID" in os.environ
HOME_PATH = os.path.expanduser("~")
PLATFORM = sys.platform  # win32, darwin, linux
FFMPEG_PATH = 'ffmpeg'