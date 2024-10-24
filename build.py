import os
import subprocess
import sys
import os
import subprocess
import sys
import shutil

import urllib.request


def python_path():
    return (
        "venv\\Scripts\\python.exe" if getPlatform() == "win32" else "venv/bin/python3"
    )


def download_file(url, destination):
    print(f"Downloading file from {url}")
    urllib.request.urlretrieve(url, destination)
    print("File downloaded successfully")


# Rest of the code...


def getPlatform():
    return sys.platform


def build_gui():
    print("Building GUI")
    if getPlatform() == "darwin" or getPlatform() == "linux":
        os.system("pyside6-uic -g python testRVEInterface.ui > mainwindow.py")
    if getPlatform() == "win32":
        os.system(
            r".\venv\Lib\site-packages\PySide6\uic.exe -g python testRVEInterface.ui > mainwindow.py"
        )


def install_pip():
    download_file("https://bootstrap.pypa.io/get-pip.py", "get-pip.py")
    command = ["python3", "get-pip.py"]
    subprocess.run(command)


def install_pip_in_venv():
    command = [
        "venv\\Scripts\\python.exe" if getPlatform() == "win32" else "venv/bin/python3",
        "get-pip.py",
    ]
    subprocess.run(command)


def build_resources():
    print("Building resources.rc")
    if getPlatform() == "darwin" or getPlatform() == "linux":
        os.system("pyside6-rcc -g python resources.qrc > resources_rc.py")
    if getPlatform() == "win32":
        os.system(
            r".\venv\Lib\site-packages\PySide6\rcc.exe -g python resources.qrc > resources_rc.py"
        )


def create_venv(python_version="python3.11"):
    print("Creating virtual environment")
    command = [python_version, "-m", "venv", "venv"]
    subprocess.run(command)


def install_requirements_in_venv():
    print("Installing requirements in virtual environment")
    command = [
        python_path(),
        "-m",
        "pip",
        "install",
        "-r",
        "requirements.txt",
    ]

    subprocess.run(command)


def build_executable():
    print("Building executable")
    if getPlatform() == "win32":
        command = [
            python_path(),
            "-m",
            "PyInstaller",
            "REAL-Video-Enhancer.py",
            "--collect-all",
            "PySide6",
            "--icon=icons/logo-v2.ico",
            "--noconfirm",
            "--noupx",
            # "--noconsole", this caused issues, maybe I can fix it later
        ]
    else:
        command = [
            python_path(),
            "-m",
            "cx_Freeze",
            "REAL-Video-Enhancer.py",
            "--target-dir",
            "dist",
        ]
    subprocess.run(command)


def clean():
    print("Cleaning up")
    os.remove("get-pip.py")


def checkIfExeExists(exe):
    path = shutil.which(exe)
    return path is not None


install_pip()
linux_and_mac_py_ver = "python3.10"
python_version = (
    linux_and_mac_py_ver
    if getPlatform() != "win32" and checkIfExeExists(linux_and_mac_py_ver)
    else "python3"
)
create_venv(python_version=python_version)
install_pip_in_venv()
install_requirements_in_venv()
build_gui()
build_resources()
if len(sys.argv) > 1:
    if sys.argv[1] == "--build_exe":
        build_executable()
