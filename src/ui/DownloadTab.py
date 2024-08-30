from PySide6.QtWidgets import QMainWindow
from .QTcustom import RegularQTPopup, NetworkCheckPopup
from ..DownloadDeps import DownloadDependencies
from ..ModelHandler import downloadAllModels

class DownloadTab:
    def __init__(
        self,
        parent: QMainWindow,
    ):
        self.parent = parent
        self.downloadDeps = DownloadDependencies()
        self.QButtonConnect()

    def QButtonConnect(self):
        self.parent.downloadNCNNBtn.clicked.connect(lambda: self.download("ncnn"))
        self.parent.downloadTorchCUDABtn.clicked.connect(
            lambda: self.download("torch_cuda")
        )
        self.parent.downloadTensorRTBtn.clicked.connect(
            lambda: self.download("tensorrt")
        )
        self.parent.downloadTorchROCmBtn.clicked.connect(
            lambda: self.download("torch_rocm")
        )
        self.parent.downloadAllModelsBtn.clicked.connect(
            downloadAllModels
        )
    def download(self, dep):
        """
        Downloads the specified dependency.
        Parameters:
        - dep (str): The name of the dependency to download.
        Returns:
        - None
        """
        if NetworkCheckPopup("https://pypi.org/"): # check for network before installing
            match dep:
                case "ncnn":
                    self.downloadDeps.downloadNCNNDeps()
                case "torch_cuda":
                    self.downloadDeps.downloadPyTorchCUDADeps()
                case "tensorrt":
                    self.downloadDeps.downloadTensorRTDeps()
                case "torch_rocm":
                    self.downloadDeps.downloadPyTorchROCmDeps()
            RegularQTPopup("Download Complete\nPlease restart the application to apply changes.")
        