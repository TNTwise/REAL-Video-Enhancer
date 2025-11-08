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
        from .ui.QTcustom import SettingUpBackendPopup, TextOutputPopup

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
        if "ERROR" in output or "TRACEBACK" in output or return_code == "1":
            TextOutputPopup(f"ERROR DETECTED IN BACKEND SETUP!\n{output}", title="FATAL ERROR")
            exit(1)
        
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
