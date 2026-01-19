import requests
import re
import os
import subprocess
import shutil

from .QTcustom import DownloadProgressPopup, NetworkCheckPopup, RegularQTPopup
from ..constants import (
    PYTHON_EXECUTABLE_PATH,
    PYTHON_DIRECTORY,
    BACKEND_PATH,
    EXE_NAME,
    LIBS_PATH,
    EXE_PATH,
    PLATFORM,
    TEMP_DOWNLOAD_PATH,
    LIBS_NAME,
    PYTHON_VERSION,
    HAS_NETWORK_ON_STARTUP,
)
from ..DownloadDeps import DownloadDependencies
from ..version import version, backend_dev_version
from ..Util import FileHandler, networkCheck, log

# version = "2.1.0" # for debugging


class PythonUpdater:
    def __init__(self):
        self.current_python_version = self.get_current_python_version()
        self.deps = DownloadDependencies()

    def get_current_python_version(self):
        try:
            output = subprocess.run(
                [PYTHON_EXECUTABLE_PATH, "--version"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:  # if python is not found
            RegularQTPopup("Python not found! Downloading Python...")
            self.deps.downloadPython(mode="Downloading")
        output = output.stdout.strip().split(" ")[
            1
        ]  # this extracts the version number from the output
        return output

    def is_python_up_to_date(self):
        log(f"Python up to date: {PYTHON_VERSION == self.current_python_version}")
        return PYTHON_VERSION == self.current_python_version

    def update_python(self):
        if HAS_NETWORK_ON_STARTUP:
            FileHandler.removeFolder(
                PYTHON_DIRECTORY
            )  # remove the old python directory
            os.mkdir(PYTHON_DIRECTORY)  # create a new python directory
            self.deps.downloadPython(mode="Updating")
            self.current_python_version = self.get_current_python_version()
        else:
            RegularQTPopup(
                "No network connection found! Please connect to the internet to update Python."
            )


class BackendUpdater:
    def __init__(self):
        self.deps = DownloadDependencies()
        self.backend_version = self.get_backend_version()

    def get_backend_version(self, iter=0):
        try:
            output = subprocess.run(
                [
                    PYTHON_EXECUTABLE_PATH,
                    os.path.join(BACKEND_PATH, "rve-backend.py"),
                    "--version",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            output = (
                output.stdout.strip()
            )  # this extracts the version number from the output
            log(f"Backend Version: {output}")
            return output
        except subprocess.CalledProcessError:  # if the backend is not found
            self.deps.downloadBackend()
            return None

    def is_backend_up_to_date(self):
        _s = backend_dev_version == self.backend_version
        log(f"Backend up to date: {_s} {backend_dev_version} == {self.backend_version}")
        return backend_dev_version == self.backend_version

    def update_backend(self):
        if HAS_NETWORK_ON_STARTUP:
            FileHandler.removeFolder(BACKEND_PATH)  # remove the old backend directory
            self.deps.downloadBackend()
        else:
            RegularQTPopup(
                "No network connection found! Please connect to the internet to update the backend."
            )


class ApplicationUpdater:
    def __init__(self):
        if HAS_NETWORK_ON_STARTUP:
            self.tag = self.get_latest_version_tag()
            self.clean_tag = self.get_latest_version_tag(clean_tag=True)
            if PLATFORM == "win32":
                platform_name = "Windows"
            elif PLATFORM == "darwin":
                platform_name = "MacOS"
            else:
                platform_name = "Linux"

            self.file_name = f"REAL-Video-Enhancer-{self.clean_tag}-{platform_name}.zip"
            self.download_url = self.build_download_url()

    def check_for_updates(self):
        if networkCheck():
            tag = self.get_latest_version_tag(clean_tag=True)
            return not tag == version and int(tag.replace(".", "")) > int(
                version.replace(".", "")
            )  # returns true if there is a new version
        return False

    def download_new_version(self):
        full_download_path = os.path.join(TEMP_DOWNLOAD_PATH, self.file_name)
        DownloadProgressPopup(
            link=self.download_url,
            downloadLocation=full_download_path,
            title=f"Downloading {self.tag}",
        )
        FileHandler.unzipFile(
            os.path.join(TEMP_DOWNLOAD_PATH, self.file_name), TEMP_DOWNLOAD_PATH
        )

    def remove_old_files(self):
        FileHandler.removeFolder(BACKEND_PATH)
        FileHandler.removeFolder(LIBS_PATH)
        FileHandler.removeFile(EXE_PATH)

    def move_new_files(self):
        if PLATFORM == "linux":
            folder_to_copy_from = "bin"
        elif PLATFORM == "win32":
            folder_to_copy_from = "REAL-Video-Enhancer"
        else:
            folder_to_copy_from = "REAL-Video-Enhancer"

        FileHandler.moveFolder(
            os.path.join(TEMP_DOWNLOAD_PATH, folder_to_copy_from, "backend"),
            os.path.join(BACKEND_PATH),
        )
        FileHandler.moveFolder(
            os.path.join(TEMP_DOWNLOAD_PATH, folder_to_copy_from, EXE_NAME),
            os.path.join(EXE_PATH),
        )
        FileHandler.moveFolder(
            os.path.join(TEMP_DOWNLOAD_PATH, folder_to_copy_from, LIBS_NAME),
            os.path.join(LIBS_PATH),
        )

    def make_exe_executable(self):
        FileHandler.makeExecutable(EXE_PATH)

    def build_download_url(self):
        url = f"https://github.com/tntwise/real-video-enhancer/releases/download/{self.tag}/{self.file_name}"
        return url

    def get_latest_version_tag(self, clean_tag=False) -> str:
        url = "https://api.github.com/repos/tntwise/real-video-enhancer/releases/latest"
        response = requests.get(url)
        if response.status_code == 200:
            latest_release = response.json()
            tag_name = latest_release["tag_name"]

        else:
            print(f"Failed to fetch latest version: {response.status_code}")
            tag_name = version  # return current version if it failed to get the latest versions

        if clean_tag:
            tag_name = re.sub(r"[^0-9.]", "", tag_name)

        return tag_name

    def install_new_update(self):
        if NetworkCheckPopup():
            if self.check_for_updates():
                self.download_new_version()
                self.remove_old_files()
                self.move_new_files()
                self.make_exe_executable()
                RegularQTPopup("Update complete! Please restart the app.")
            else:
                RegularQTPopup("No update available!")


class DependencyUpdateChecker:
    def __init__(self, installed_dependencies: list[str]):
        self.installed_dependencies = installed_dependencies

    def getPipVersion(self, dependency):
        command = [
            PYTHON_EXECUTABLE_PATH,
        ]

    def getDepVers(self):
        for dependency in self.installed_dependencies:
            if "pytorch (rocm)" in dependency:
                pass


if __name__ == "__main__":
    updater = ApplicationUpdater()
    print(updater.check_for_updates())
    print(updater.build_download_url())
