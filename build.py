import os
import subprocess
import sys
def getPlatform():
    return sys.platform
        


def build_gui():
    if getPlatform() == "darwin" or getPlatform() == "linux":
        os.system("pyside6-uic -g python testRVEInterface.ui > mainwindow.py")
    if getPlatform() == "win32":
        os.system(
            r".\venv\Lib\site-packages\PySide6\uic.exe -g python testRVEInterface.ui > mainwindow.py"
        )


def build_resources():
    if getPlatform() == "darwin" or getPlatform() == "linux":
        os.system("pyside6-rcc -g python resources.qrc > resources_rc.py")
    if getPlatform() == "win32":
        os.system(
            r".\venv\Lib\site-packages\PySide6\rcc.exe -g python resources.qrc > resources_rc.py"
        )

def create_venv():
    print("Creating virtual environment")
    command = ["python3", "-m", "venv", "venv"]
    subprocess.run(command)

def install_requirements_in_venv():
    if getPlatform() == "win32":
        print("Installing requirements in windows virtual environment")
        command = [
                "venv\\Scripts\\python.exe",
                "-m",
                "pip",
                "install",
                "-r",
                "requirements.txt",
            ]
        subprocess.run(command)
    


def build_executable():
    if getPlatform() == "linux" or getPlatform() == "darwin": 
        command = [
            "python3",
            "-m",
            "PyInstaller",
            "main.py",
            "--noconfirm",
            "--noupx",
        ]
        subprocess.run(command)
    if getPlatform() == "win32":
        command = [
            r".\venv\Scripts\python.exe",
            "-m",
            "PyInstaller",
            "main.py",
            "--noconfirm",
            "--noupx",
        ]

        subprocess.run(command)
        # copy("backend","dist\\main\\backend")


create_venv()
install_requirements_in_venv()
build_gui()
build_resources()
if len(sys.argv) > 1:
    if sys.argv[1] == "--build_exe":
        build_executable()
