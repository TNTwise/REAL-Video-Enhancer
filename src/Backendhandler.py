import os

from .constants import BACKEND_PATH, PYTHON_EXECUTABLE_PATH, PYTHON_DIRECTORY, PLATFORM, IS_INSTALLED, IS_FLATPAK, HAS_NETWORK_ON_STARTUP, CWD
from .Util import (
    FileHandler
)
from PySide6.QtWidgets import QMessageBox
from .version import version


class BackendHandler:
    def __init__(self, parent, settings=None):
        self.parent = parent
        self.settings = settings

    def getAvailableBackends(self):
        from .ui.QTcustom import SettingUpBackendPopup, RegularQTPopup

        output = SettingUpBackendPopup(
            [
                PYTHON_EXECUTABLE_PATH,
                "-W",
                "ignore",
                os.path.join(BACKEND_PATH, "rve-backend.py"),
                "--list_backends",
            ]
        )
        return_code = str(output.getReturnCode()).strip()
        output: str = output.getOutput()
        
        """if return_code == "1":
            reply = QMessageBox.question(
                self.parent,
                "",
                f"Getting available backends failed!\nDelete {PYTHON_DIRECTORY} and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,  # type: ignore
            )
            if reply == QMessageBox.Yes:  # type: ignore
                FileHandler.removeFolder(PYTHON_DIRECTORY)
                os._exit(0)"""
        output = output.split(" ")
        # hack to filter out bad find
        new_out = ""
        for word in output:
            if "objc" in word:
                continue
            if "[Torch-TensorRT]" in word:
                continue
            new_out += word + " "

        # Find the part of the output containing the backends list
        output = new_out
        start = output.find("[")
        end = output.find("]") + 1
        backends_str = output[start:end]

        # Convert the string representation of the list to an actual list
        try:
            backends = eval(backends_str)
        except Exception:
            backends = []

        return backends, output
