import os
import subprocess
from src.Util import getPlatform

def build_gui():
    if getPlatform() == "linux":
        os.system("pyside6-uic -g python testRVEInterface.ui > mainwindow.py")
    if getPlatform() == "win32":
        os.system(r".\venv\Lib\site-packages\PySide6\uic.exe -g python testRVEInterface.ui > mainwindow.py")

def build_resources():
    if getPlatform() == "linux":
        os.system("pyside6-rcc -g python resources.qrc > resources_rc.py")
    if getPlatform() == "win32":
        os.system(r".\venv\Lib\site-packages\PySide6\rcc.exe -g python resources.qrc > resources_rc.py")

build_gui()
build_resources()
