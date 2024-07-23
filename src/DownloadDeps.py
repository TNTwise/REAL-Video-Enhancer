from .Util import (
    getPlatform,
    printAndLog,
    pythonPath,
    ffmpegPath,
    currentDirectory,
    createDirectory,
    removeFile,
    makeExecutable
)
from .QTcustom import DownloadProgressPopup
import requests
import os
import tarfile
from threading import Thread

class DownloadDependencies:
    """
    Downloads platform specific dependencies python and ffmpeg to their respective locations and creates the directories

    """

    def __init__(self):
        createDirectory(os.path.join(currentDirectory(), "python"))
        createDirectory(os.path.join(currentDirectory(), "ffmpeg"))

    def extractTarGZ(self, file):
        """
        Extracts a tar gz in the same directory as the tar file and deleted it after extraction.
        """
        printAndLog("Extracting: " + file)
        tar = tarfile.open(file, "r:gz")
        tar.extractall()
        tar.close()
        removeFile(file)

    def downloadPython(self):
        pythonInstallLocation = os.path.join(
            currentDirectory(), "python", "python.tar.gz"
        )
        match getPlatform():
            case "linux":
                link = "https://github.com/indygreg/python-build-standalone/releases/download/20240713/cpython-3.11.9+20240713-x86_64-unknown-linux-gnu-install_only.tar.gz"
            case "win32":
                link = "https://github.com/indygreg/python-build-standalone/releases/download/20240713/cpython-3.11.9+20240713-x86_64-pc-windows-msvc-install_only.tar.gz"
        # probably can add macos support later
        printAndLog("Downloading Python")
        DownloadProgressPopup(link=link, downloadLocation=pythonInstallLocation)

        # extract python
        self.extractTarGZ(pythonInstallLocation)
        makeExecutable(pythonPath())

    def downloadFFMpeg(self):

        match getPlatform():
            case "linux":
                link = "https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/ffmpeg"
            case "win32":
                link = "https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/ffmpeg.exe"

        printAndLog("Downloading FFMpeg")
        DownloadProgressPopup(link=link, downloadLocation=ffmpegPath())
        makeExecutable(ffmpegPath())
        

if __name__ == "__main__":
    downloadDependencies = DownloadDependencies()
    downloadDependencies.downloadPython()
