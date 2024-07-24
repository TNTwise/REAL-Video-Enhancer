from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import QThread, QObject

import requests
from .QTstyle import styleSheet


class DownloadAndReportToQTThread(QThread):
    """
    Downloads a file while reporting the actual bytes downloaded
    """

    finished = QtCore.Signal()
    progress = QtCore.Signal(int)
    
    def __init__(self, link, downloadLocation,parent=None):
        super().__init__(parent)
        self.link = link
        self.downloadLocation = downloadLocation
    def run(self):
        response = requests.get(
            self.link,
            stream=True,
        )
        totalByteSize = int(response.headers["Content-Length"])
        totalSize = 0
        with open(self.downloadLocation, "wb") as f:
            chunk_size = 128
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                totalSize += chunk_size
                size = totalSize / totalByteSize * 100
                self.progress.emit(size)
        self.finished.emit()


class DownloadProgressPopup(QtWidgets.QProgressDialog):
    """
    Runs a download of a file in a thread while reporitng progress to a qt progressbar popup.
    This wont start up in a new process, rather it will just take over the current process
    """

    def __init__(self, link: str, downloadLocation: str, title: str = None):
        super().__init__()
        self.link = link
        self.downloadLocation = downloadLocation
        
        self.setWindowTitle(title)
        self.setStyleSheet(styleSheet())
        self.setRange(0, 100)
        self.setMinimumSize(300, 100)
        self.setMaximumSize(300, 100)
        self.startDownload()
        self.exec()

    """
    Initializes all threading bs
    """

    def startDownload(self):
        self.workerThread = DownloadAndReportToQTThread(link=self.link, downloadLocation=self.downloadLocation)
        self.workerThread.progress.connect(self.setProgress)
        self.workerThread.finished.connect(self.workerThread.deleteLater)
        self.workerThread.finished.connect(self.close)
        self.workerThread.start()

    def setProgress(self, value):
        self.setValue(value)


if __name__ == '__main__':
    DownloadProgressPopup(
            link="https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/ffmpeg", downloadLocation="ffmpeg", title="Downloading Python"
        )