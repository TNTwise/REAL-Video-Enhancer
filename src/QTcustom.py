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

    def __init__(self, parent=None):
        QThread.__init__(self, parent)

    def run(self, link, downloadLocation):
        response = requests.get(
            link,
            stream=True,
        )
        totalSize = 0
        with open(downloadLocation, "wb") as f:
            chunk_size = 128
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                totalSize += chunk_size
                self.progress.emit(totalSize)
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
        self.workerThread = QThread()
        self.worker = DownloadAndReportToQTThread()
        totalSize = int(requests.get(link).headers["Content-Length"])
        self.setWindowTitle(title)
        self.setStyleSheet(styleSheet())
        self.setRange(0, totalSize)
        self.setMinimumSize(300, 100)
        self.setMaximumSize(300, 100)
        self.startDownload()
        self.exec()

    """
    Initializes all threading bs
    """

    def startDownload(self):
        self.worker.moveToThread(self.workerThread)
        self.workerThread.started.connect(
            lambda: self.worker.run(
                link=self.link, downloadLocation=self.downloadLocation
            )
        )
        self.worker.progress.connect(self.setProgress)
        self.worker.finished.connect(self.workerThread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.workerThread.finished.connect(self.workerThread.deleteLater)
        self.workerThread.finished.connect(self.close)
        self.workerThread.start()

    def setProgress(self, value):
        self.setValue(value)
