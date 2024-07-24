from .Util import (
    getPlatform,
    printAndLog,
    pythonPath,
    ffmpegPath,
    currentDirectory,
    createDirectory,
    removeFile,
    makeExecutable,
)
from .QTcustom import DownloadProgressPopup
import os
import tarfile


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
        link = "https://github.com/indygreg/python-build-standalone/releases/download/20240713/cpython-3.11.9+20240713-+20240713-"
        match getPlatform():
            case "linux":
                link += "x86_64-unknown-linux-gnu-install_only.tar.gz"
            case "win32":
                link += "x86_64-pc-windows-msvc-install_only.tar.gz"
        # probably can add macos support later
        printAndLog("Downloading Python")
        DownloadProgressPopup(link=link, downloadLocation=pythonPath(),title="Downloading Python")

        # extract python
        self.extractTarGZ(pythonPath())
        makeExecutable(pythonPath())

    def downloadFFMpeg(self):
        link = "https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/"
        match getPlatform():
            case "linux":
                link += "ffmpeg"
            case "win32":
                link += "ffmpeg.exe"

        printAndLog("Downloading FFMpeg")
        DownloadProgressPopup(link=link, downloadLocation=ffmpegPath(), title="Downloading FFMpeg")
        makeExecutable(ffmpegPath())



if __name__ == "__main__":
    downloadDependencies = DownloadDependencies()
    downloadDependencies.downloadPython()
