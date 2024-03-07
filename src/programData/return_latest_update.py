from src.programData.checks import *
import requests
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot


def return_latest_git_ver(url):
    latest = requests.get(url)
    latest = latest.url
    return latest.split("/")[-1]


class return_latest(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(list)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)

    def run(self):
        print("h")
        if check_if_online(True):
            try:
                latest_beta = return_latest_git_ver(
                    "https://github.com/TNTwise/REAL-Video-Enhancer-BETA/releases/latest"
                )
                latest_stable = return_latest_git_ver(
                    "https://github.com/TNTwise/REAL-Video-Enhancer/releases/latest"
                )

                self.progress.emit([latest_beta, latest_stable])
                self.finished.emit()
            except:
                self.finished.emit()
                pass
        else:
            self.finished.emit()
            print("why")
            return "why"
