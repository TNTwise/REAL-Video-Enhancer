import os
from PySide6.QtWidgets import QMainWindow, QMessageBox
from .QTcustom import RegularQTPopup, NetworkCheckPopup, remove_combobox_item_by_text
from ..DownloadDeps import DownloadDependencies
from .Updater import ApplicationUpdater
from ..constants import IS_FLATPAK, PLATFORM, CWD, USE_LOCAL_BACKEND, HOME_PATH, PLATFORM, IS_FLATPAK, CWD, CPU_ARCH
from ..BuiltInTorchVersions import TorchVersion
from ..Util import FileHandler


class DownloadTab:
    def __init__(
        self,
        parent: QMainWindow,
        backends: list,
    ):
        self.parent = parent
        self.torch_versions:list[TorchVersion] = [version for version in TorchVersion.__subclasses__()]
        self.downloadDeps = DownloadDependencies()
        self.backends = backends
        self.applicationUpdater = ApplicationUpdater()


        if IS_FLATPAK:
            minimum_space_required = 7
        else:
            minimum_space_required = 15
        self.parent.low_storage_label.setVisible(False)
        if FileHandler.getFreeSpace() < minimum_space_required:
            self.parent.downloadTorchBtn.setEnabled(False)
            self.parent.downloadTensorRTBtn.setEnabled(False)
            self.parent.low_storage_label.setVisible(True)

        # disable as it is not complete
        try:
            self.parent.downloadDirectMLBtn.setEnabled(False)
            if PLATFORM != "win32":
                self.parent.downloadDirectMLBtn.setEnabled(False)
        except Exception as e:
            print(e)


        # set this all to not visible, as scrapping the idea for now.
        if PLATFORM != "linux":
            remove_combobox_item_by_text(self.parent.pytorch_backend, "ROCm")
        else:
            if CPU_ARCH == "arm64":
                remove_combobox_item_by_text(self.parent.pytorch_backend, "XPU")
                remove_combobox_item_by_text(self.parent.pytorch_backend, "ROCm")
        
        if PLATFORM == "darwin":
            if CPU_ARCH == "arm64":
                self.parent.pytorch_backend.clear()
                self.parent.pytorch_backend.addItems(
                    ["MPS (Apple Silicon)"]
                )
                # force 2.9.0 as it should include support for uint16
                self.parent.pytorch_version.setEnabled(False)

                self.parent.pytorch_backend.setCurrentText("MPS (Apple Silicon)")
                self.parent.pytorch_backend.setEnabled(False)
                
                self.parent.downloadTorchBtn.setEnabled(True)
            self.parent.downloadTensorRTBtn.setEnabled(False)
        if IS_FLATPAK or USE_LOCAL_BACKEND:
            self.parent.uninstallAppBtn.setDisabled(True)
        else:
            self.parent.uninstallAppBtn.clicked.connect(self.uninstallApp)

        self.parent.ApplicationUpdateContainer.setVisible(False)
        self.QButtonConnect()
    
    def QButtonConnect(self):
        self.parent.downloadNCNNBtn.clicked.connect(lambda: self.download("ncnn", True))
        self.parent.downloadTorchBtn.clicked.connect(
            lambda: self.download("torch", True)
        )
        self.parent.downloadTensorRTBtn.clicked.connect(
            lambda: self.download("tensorrt", True)
        )
        self.parent.downloadDirectMLBtn.clicked.connect(
            lambda: self.download("directml", True)
        )
        
        self.parent.uninstallNCNNBtn.clicked.connect(
            lambda: self.download("ncnn", False)
        )
        self.parent.uninstallTorchBtn.clicked.connect(
            lambda: self.download("torch", False)
        )
        self.parent.uninstallTensorRTBtn.clicked.connect(
            lambda: self.download("tensorrt", False)
        )
        self.parent.uninstallDirectMLBtn.clicked.connect(
            lambda: self.download("directml", False)
        )
        self.parent.selectPytorchCustomModel.clicked.connect(
            lambda: self.parent.importCustomModel("pytorch")
        )
        self.parent.selectNCNNCustomModel.clicked.connect(
            lambda: self.parent.importCustomModel("ncnn")
        )
        self.parent.pytorch_version.addItems(
            [version.torch_version for version in self.torch_versions]
        )

    def uninstallApp(self):
        reply = QMessageBox.question(
            self.parent,
            "",
            "Are you sure you want to uninstall?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,  # type: ignore
        )
        if reply == QMessageBox.Yes:  # type: ignore
            os.chdir(HOME_PATH) # fix for windows, as you cant delete a directory in use.
            FileHandler().removeFolder(CWD)
            os._exit(0)
        

        
    def hideUninstallButtons(self):
        self.parent.uninstallTorchBtn.setVisible(False)
        self.parent.uninstallNCNNBtn.setVisible(False)
        self.parent.uninstallTensorRTBtn.setVisible(False)
        self.parent.uninstallDirectMLBtn.setVisible(False)

    def showUninstallButton(self, backends):
        if "pytorch (cuda)" in backends:
            self.parent.downloadTorchBtn.setVisible(False)
            self.parent.uninstallTorchBtn.setVisible(True)
        if "pytorch (rocm)" in backends:
            self.parent.downloadTorchBtn.setVisible(False)
            self.parent.uninstallTorchBtn.setVisible(True)
        if "pytorch (xpu)" in backends:
            self.parent.downloadTorchBtn.setVisible(False)
            self.parent.uninstallTorchBtn.setVisible(True)
        if "pytorch (mps)" in backends:
            self.parent.downloadTorchBtn.setVisible(False)
            self.parent.uninstallTorchBtn.setVisible(True)
        if "ncnn" in backends:
            self.parent.downloadNCNNBtn.setVisible(False)
            self.parent.uninstallNCNNBtn.setVisible(True)
        if "tensorrt" in backends:
            self.parent.downloadTensorRTBtn.setVisible(False)
            self.parent.uninstallTensorRTBtn.setVisible(True)

        # disable as it is not complete
        try:
            self.parent.downloadDirectMLBtn.setEnabled(False)
            if PLATFORM != "win32":
                self.parent.downloadDirectMLBtn.setEnabled(False)
        except Exception as e:
            print(e)


    def download(self, dep, install: bool = True):
        """
        Downloads the specified dependency.
        Parameters:
        - dep (str): The name of the dependency to download.
        Returns:
        - None
        """
        if install and ("torch" in dep.lower() or "tensorrt" in dep.lower()):
            if PLATFORM != "darwin":
                reply = QMessageBox.question(
                    self.parent,
                    "",
                    "Old GTX cards require torch version 2.6.0.\nContinue installation?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,  # type: ignore
                )
                if reply == QMessageBox.Yes:  # type: ignore
                    pass
                else:
                    return
        pytorch_ver:TorchVersion|None = None
        current_pytorch_version = self.parent.pytorch_version.currentText().split()[0]
        current_pytorch_backend = self.parent.pytorch_backend.currentText().split()[0].lower()
        for version in self.torch_versions:
            if version.torch_version == current_pytorch_version:
                pytorch_ver = version
                torchvision_ver = version.torchvision_version

        if not pytorch_ver:
            RegularQTPopup(
                "Please select a valid PyTorch version from the dropdown."
            )
            return
        
        if current_pytorch_backend == "cuda" or dep.lower() == "tensorrt":
            pytorch_backend = pytorch_ver.cuda_version
        elif current_pytorch_backend == "rocm":
            pytorch_backend = pytorch_ver.rocm_version
        elif current_pytorch_backend == "xpu":
            pytorch_backend = pytorch_ver.xpu_version
        elif current_pytorch_backend == "mps":
            pytorch_backend = pytorch_ver.mps_version
        
        if NetworkCheckPopup(
            "https://pypi.org/"
        ):  # check for network before installing
            return_code = self.downloadDeps.downloadPythonDeps(dep, pytorch_ver.torch_version, torchvision_ver, pytorch_backend, install)
            if return_code == 0:
                RegularQTPopup(
                    "Download Complete\nPlease restart the application to apply changes."
                )
            else:
                RegularQTPopup("Download Failed!\nPlease check logs for more info.")
