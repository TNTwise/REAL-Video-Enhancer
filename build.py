from abc import  abstractmethod
import os
import subprocess
import sys
import os
import subprocess
import sys
import shutil
import argparse
import requests
import urllib.request

PLATFORM = sys.platform
OUTPUT_FOLDER = "dist"

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
                width.text = "0"
            if height is not None:
                height.text = "0"
            tree.write(path)

    set_mainwindow_size_zero()

def downloadFile(link, downloadLocation):
    response = requests.get(
        link,
        stream=True,
    )

    with open(downloadLocation, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)


class PythonManager:

    PYTHON_VENV_PATH = "venv\\Scripts\\python.exe" if PLATFORM == "win32" else "venv/bin/python3"
    PYTHON_SYSTEM_EXECUTABLE = "python3"

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
    
    def download_file(self, url, destination):
        print(f"Downloading file from {url}")
        urllib.request.urlretrieve(url, destination)
        print("File downloaded successfully")

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

    
    def copy_backend(self, build_dir=None):
        print("Copying backend")
        if PLATFORM == "win32":
            if build_dir is None:
                build_dir = "dist"
            try:
                os.system(f"cp -r backend {build_dir}/REAL-Video-Enhancer/backend")
            except Exception:
                pass
            if not os.path.exists(rf"{build_dir}\\REAL-Video-Enhancer\\backend"):
                os.system(
                    f'xcopy "./backend" "./{build_dir}/REAL-Video-Enhancer/backend" /E /I'
                )
        if PLATFORM == "linux":
            if build_dir is None:
                build_dir = "bin"
            os.system(f"cp -r backend {build_dir}/")


class PyInstaller(BuildManager):
    pyinstaller_version = "pyinstaller==6.12.0"

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
            + " --noconsole"
            + " --distpath"
            + f" {OUTPUT_FOLDER}"
            )
        )
            
class CxFreeze(BuildManager):

    cx_freeze_version = "cx_freeze==7.0.0"

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
        if PLATFORM == "linux":
            try:
                if not os.path.isfile('/usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0'):
                    raise FileNotFoundError("Unable to build as libxcbcursor is not installed!")
                
                input_file = "/usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0"
            except FileNotFoundError:
                try:
                    print("libxcbcursor not found, downloading...")
                    
                    downloadFile("https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/libxcb-cursor.so.0","libxcb-cursor.so.0")
                    input_file = "libxcb-cursor.so.0"
                except requests.exceptions.ConnectionError:
                    raise FileNotFoundError("libxcbcursor not installed, and no network available to download it!")
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

if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument("--build", help="Build the application with a specific builder.", default="gui", choices=["pyinstaller", "cx_freeze", "nuitka", "gui"])
    args.add_argument("--copy_backend", help="Copy the backend to the build directory", action="store_true")    
    args = args.parse_args()
    BuildManager().python_manager.pip_install_package_in_venv("PySide6==6.8.0") # up to date pyside6 version to build the GUI
    BuildManager().build_resources()
    BuildManager().build_gui()
    BuildManager().python_manager.setup_python()
    if PLATFORM is not 'win32':
        BuildManager().python_manager.pip_install_package_in_venv("PySide6==6.8.0") # linux/mac not impacted

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
    if args.copy_backend:
        builder.copy_backend()
    print("Build complete")

    