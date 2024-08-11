import os
import subprocess
from src.Util import getPlatform

def setup_windows():
    if getPlatform() == "win32":
        command = ["python3.10.exe","-m","venv","venv"]
        subprocess.run(command)
        command = ["venv\\Scripts\\python.exe","-m","pip","install","-r","requirements.txt","--force","--no-cache-dir"]
        subprocess.run(command)    

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

def build_executable():
    if getPlatform() == "linux":
        command = ["python3","-m","cx_Freeze","main.py"]
        subprocess.run(command)
    if getPlatform() == "win32":
        command = [r".\venv\Scripts\python.exe","-m","PyInstaller","main.py"]
        subprocess.run(command)

setup_windows()
build_gui()
build_resources()
build_executable()