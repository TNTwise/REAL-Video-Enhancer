from PySide6.QtWidgets import QMainWindow
from .QTcustom import RegularQTPopup, NetworkCheckPopup
from ..DownloadDeps import DownloadDependencies
from ..ModelHandler import downloadModelsBasedOnInstalledBackend


class DownloadTab:
    def __init__(
        self,
        parent: QMainWindow,
        installed_backends: list,
    ):
        self.parent = parent
        self.installed_backends = installed_backends
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
        self.parent.downloadDirectMLBtn.clicked.connect(
            lambda: self.download("directml")
        )
        self.parent.downloadAllModelsBtn.clicked.connect(
            lambda: downloadModelsBasedOnInstalledBackend(
                ["ncnn", "pytorch", "tensorrt", "directml"]
            )
        )
        self.parent.downloadSomeModelsBasedOnInstalledBackendbtn.clicked.connect(
            lambda: downloadModelsBasedOnInstalledBackend(self.installed_backends)
        )

    def download(self, dep):
        """
        Downloads the specified dependency.
        Parameters:
        - dep (str): The name of the dependency to download.
        Returns:
        - None
        """
        if NetworkCheckPopup(
            "https://pypi.org/"
        ):  # check for network before installing
            match dep:
                case "ncnn":
                    self.downloadDeps.downloadNCNNDeps()
                case "torch_cuda":
                    self.downloadDeps.downloadPyTorchCUDADeps()
                case "tensorrt":
                    self.downloadDeps.downloadTensorRTDeps()
                case "torch_rocm":
                    self.downloadDeps.downloadPyTorchROCmDeps()
                case "directml":
                    self.downloadDeps.downloadDirectMLDeps()
            RegularQTPopup(
                "Download Complete\nPlease restart the application to apply changes."
            )
