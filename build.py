from abc import  abstractmethod
import os
import subprocess
import sys
import os
import subprocess
import sys
import shutil
import argparse
import platform
import urllib.request

PLATFORM = sys.platform
CPU_ARCH = "x86_64" if platform.machine() == "AMD64" else platform.machine()
OUTPUT_FOLDER = "dist"
print(f"Platform: {PLATFORM}")
print(f"CPU Arch: {CPU_ARCH}")
print(f"OUTPUT_FOLDER: {OUTPUT_FOLDER}")


def zero_mainwindow_size():
    import xml.etree.ElementTree as ET

    def set_mainwindow_size_zero(path="testRVEInterface.ui"):
        tree = ET.parse(path)
        root = tree.getroot()

        geometry = root.find('.//property[@name="geometry"]/rect')
        if geometry is not None:
            width = geometry.find("width")
            height = geometry.find("height")
            if width is not None:
                width.text = "1000"
            if height is not None:
                height.text = "700"
            tree.write(path)

    set_mainwindow_size_zero()


def download_file(url, destination):
        print(f"Downloading file from {url}")
        urllib.request.urlretrieve(url, destination)
        print("File downloaded successfully")

def get_libxcb_cursor_binary():
    try:
        if CPU_ARCH == "x86_64":
            input_file = '/usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0'
        else:
            input_file = '/usr/lib/aarch64-linux-gnu/libxcb-cursor.so.0'
        if not os.path.isfile(input_file):
            raise FileNotFoundError("Unable to build as libxcbcursor is not installed!")
        
    except FileNotFoundError:
        try:
            print("libxcbcursor not found, downloading...")
            if CPU_ARCH == "x86_64":
                download_file("https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/libxcb-cursor.so.0","libxcb-cursor.so.0")
            else:
                download_file("https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/libxcb-cursor.so.0_arm64","libxcb-cursor.so.0")
            input_file = "libxcb-cursor.so.0"
        except Exception as e:
            print(e)
            raise FileNotFoundError("libxcbcursor not installed, and no network available to download it!")
    return input_file
    
class PythonManager:

    PYTHON_VENV_PATH = "venv\\Scripts\\python.exe" if PLATFORM == "win32" else "venv/bin/python3"
    PYTHON_SYSTEM_EXECUTABLE = sys.executable

    def __init__(self):
        if not os.path.exists("venv"):
            self.setup_python()
    
    @classmethod
    def run_venv_python(cls, command: str):
        command = [cls.PYTHON_VENV_PATH,] + command.split()
        subprocess.run(command)
    
    @classmethod
    def pip_install_package_in_venv(cls, package: str):
        command = [
            cls.PYTHON_VENV_PATH,
            "-m",
            "pip",
            "install",
            package,
        ]
        subprocess.run(command)

    def setup_python(self):
        self.__create_venv()
        self.__install_pip_in_venv()
        self.__install_requirements_in_venv()

    def __create_venv(self):
        print("Creating virtual environment")
        command = [self.PYTHON_SYSTEM_EXECUTABLE, "-m", "venv", "venv"]
        subprocess.run(command)


    def __install_pip_in_venv(self):
        command = [
            self.PYTHON_VENV_PATH,
            "-m",
            "ensurepip",
        ]
        subprocess.run(command)

    def __install_requirements_in_venv(self):
        print("Installing requirements in virtual environment")
        if not os.path.isfile("requirements.txt"):
            raise FileNotFoundError("No requirements.txt in current directory!")
        command = [
            self.PYTHON_VENV_PATH,
            "-m",
            "pip",
            "install",
            "-r",
            "requirements.txt",
        ]

        subprocess.run(command)
        

    def get_venv_site_packages(self):
        command = [
            self.PYTHON_VENV_PATH,
            "-c",
            'import site; print("\\n".join(site.getsitepackages()))',
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
        site_packages = result.stdout.strip().split('\n')[0]
        if os.path.exists(site_packages):
            return site_packages
        site_packages = site_packages.replace('dist','site')
        if os.path.exists(site_packages):
            return site_packages
        print(site_packages)
        raise FileNotFoundError("Unable to locate site packages for python venv!")
    


class BuildManager:
    def __init__(self):
        shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
        self.python_manager = PythonManager()

    @abstractmethod
    def build(self):
        ...
    
    def build_gui(self):
        print("Building GUI")
        zero_mainwindow_size()
        if PLATFORM == "darwin" or PLATFORM == "linux":
            os.system(
                f"{self.python_manager.get_venv_site_packages()}/PySide6/Qt/libexec/uic -g python testRVEInterface.ui > mainwindow.py"
            )
        if PLATFORM == "win32":
            os.system(
                r".\venv\Lib\site-packages\PySide6\uic.exe -g python testRVEInterface.ui > mainwindow.py"
            )
    
    def build_resources(self):
        print("Building resources.rc")
        if PLATFORM == "darwin" or PLATFORM == "linux":
            os.system(
                f"{self.python_manager.get_venv_site_packages()}/PySide6/Qt/libexec/rcc -g python resources.qrc > resources_rc.py"
            )
        if PLATFORM == "win32":
            os.system(
                r".\venv\Lib\site-packages\PySide6\rcc.exe -g python resources.qrc > resources_rc.py"
            )

    
    def copy_backend(self):
        print("Copying backend")
        if "pyinstaller" in self.__str__().lower():
            backend_dir = os.path.join(f"{OUTPUT_FOLDER}/REAL-Video-Enhancer/backend")
            try:
                shutil.copytree("backend",backend_dir)
            except Exception:
                raise FileNotFoundError("Backend failed to copy!")
            if not os.path.exists(backend_dir):
                raise FileNotFoundError("Backend failed to copy!")
        else:
            shutil.copytree("backend", f"{OUTPUT_FOLDER}/backend")

    @abstractmethod
    def patch_for_xcbcursor(self):
        ...

class PyInstaller(BuildManager):
    pyinstaller_version = "pyinstaller==6.16.0"

  
    def build(self):
        print("Building executable")

        PythonManager.pip_install_package_in_venv(self.pyinstaller_version)
        PythonManager.run_venv_python(
            (
              "-m PyInstaller" 
            + " REAL-Video-Enhancer.py" 
            + " --icon=icons/logo-v2.ico" 
            + " --noconfirm"
            + " --noupx" 
            + " --noconsole" # i think this fixes weird macos dir shit
            + " --distpath"
            + f" {OUTPUT_FOLDER}"
            )
        )

    def patch_for_xcbcursor(self):
        if PLATFORM == "linux":
            input_file = get_libxcb_cursor_binary()
            print("Copying libcursor to qt lib directory")
            shutil.copy(input_file, f"{OUTPUT_FOLDER}/REAL-Video-Enhancer/_internal/PySide6/Qt/lib/")
            
class CxFreeze(BuildManager):

    cx_freeze_version = "cx_freeze==7.2.10"

    def build(self):
        print("Building executable")

        PythonManager.pip_install_package_in_venv(self.cx_freeze_version)
        PythonManager.run_venv_python(
            (
              " -m"
            + " cx_Freeze"
            + " REAL-Video-Enhancer.py"
            + " --target-dir"
            + f" {OUTPUT_FOLDER}"
            )
        )

    def patch_for_xcbcursor(self):
        if PLATFORM == "linux":
            input_file = get_libxcb_cursor_binary()
            print("Copying libcursor to qt lib directory")
            shutil.copy(input_file, f"{OUTPUT_FOLDER}/lib/PySide6/Qt/lib")
            

class Nuitka(BuildManager):

    nuitka_version = "nuitka==2.6.7"

    def build(self):
        print("Building executable")

        PythonManager.pip_install_package_in_venv(self.nuitka_version)
        PythonManager.run_venv_python(
            (
              " -m nuitka" 
            + " --standalone" 
            + " --low-memory"
            + " --include-package-data=PySide6"
            + " --include-package-data=cpuinfo"
            + " --enable-plugin=pyside6"
            + " --include-qt-plugins=qml"
            + " --show-progress" 
            + " --show-scons" 
            + f" --output-dir={OUTPUT_FOLDER}"
            + " REAL-Video-Enhancer.py"
            )
        )

    def patch_for_xcbcursor(self):
        if PLATFORM == "linux":
            raise NotImplementedError("Nuitka is not working on linux.")
            input_file = get_libxcb_cursor_binary()
            print("Copying libcursor to qt lib directory")
            shutil.copy(input_file, f"{OUTPUT_FOLDER}/lib/PySide6/Qt/lib")


if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument("--build", help="Build the application with a specific builder.", default="gui", choices=["pyinstaller", "cx_freeze", "nuitka", "gui"])
    args.add_argument("--copy_backend", help="Copy the backend to the build directory", action="store_true")    
    args = args.parse_args()
    if not os.path.exists("venv") or not args.build == "gui":
        BuildManager().python_manager.setup_python()
    BuildManager().build_resources()
    BuildManager().build_gui()
    
    match args.build:
        case "pyinstaller":
            builder = PyInstaller()
        case "cx_freeze":
            builder = CxFreeze()
        case "nuitka":
            builder = Nuitka()
        case "gui":
            exit()
        case _:
            raise ValueError("Invalid build option")
    builder.build()
    builder.patch_for_xcbcursor()
    if args.copy_backend:
        builder.copy_backend()
    print("Build complete")

    
