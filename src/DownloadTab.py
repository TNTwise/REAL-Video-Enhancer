from PySide6.QtWidgets import QMainWindow
from .DownloadDeps import DownloadDependencies


class DownloadTab:
    def __init__(
        self,
        parent: QMainWindow,
    ):
        self.parent = parent
        self.downloadDeps = DownloadDependencies()
        self.QButtonConnect()

    def QButtonConnect(self):
        self.parent.downloadNCNNBtn.clicked.connect(
            lambda: self.downloadDeps.downloadNCNNDeps()
        )
