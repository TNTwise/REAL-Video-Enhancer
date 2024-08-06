import sys
import subprocess
import requests
import time
import numpy as np
from multiprocessing import shared_memory

from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import QThread, Qt

from .QTstyle import styleSheet
from .Util import printAndLog


class UpdateGUIThread(QThread):
    """
    Gets the latest bytes outputed from the shared memory and returns them in QImage format for display
    """

    latestPreviewPixmap = Signal(QtGui.QImage)

    def __init__(
        self, parent, imagePreviewSharedMemoryID, outputVideoHeight, outputVideoWidth
    ):
        super().__init__()
        self._parent = parent
        self._stop_flag = False  # Boolean flag to control stopping
        self._mutex = QMutex()  # Atomic flag to control stopping
        self.imagePreviewSharedMemoryID = imagePreviewSharedMemoryID
        self.outputVideoHeight = outputVideoHeight
        self.outputVideoWidth = outputVideoWidth

    def run(self):
        while True:
            with QMutexLocker(self._mutex):
                if self._stop_flag:
                    break
            try:
                self.shm = shared_memory.SharedMemory(
                    name=self.imagePreviewSharedMemoryID
                )
                image_bytes = self.shm.buf[:].tobytes()

                # Convert image bytes back to numpy array
                image_array = np.frombuffer(image_bytes, dtype=np.uint8).reshape(
                    (self.outputVideoHeight, self.outputVideoWidth, 3)
                )
                pixmap = self.convert_cv_qt(image_array)
                self.latestPreviewPixmap.emit(pixmap)
            except FileNotFoundError:
                # print("preview not available")
                self.latestPreviewPixmap.emit(None)
            time.sleep(0.1)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        # rgb_image = cv2.resize(cv_img, (1280, 720)) #Cound resize image if need be
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            cv_img.data,
            w,
            h,
            bytes_per_line,
            QtGui.QImage.Format_RGB888,  # type: ignore
        )
        return convert_to_Qt_format

    def stop(self):
        with QMutexLocker(self._mutex):
            self._stop_flag = True
        try:
            self.shm.close()
            print("Closed Read Memory")
        except AttributeError as e:
            printAndLog("No read memory", str(e))  # type: ignore


# custom threads
class DownloadAndReportToQTThread(QThread):
    """
    Downloads a file while reporting the actual bytes downloaded
    """

    finished = QtCore.Signal()
    progress = QtCore.Signal(int)

    def __init__(self, link, downloadLocation, parent=None):
        super().__init__(parent)
        self.link = link
        self.downloadLocation = downloadLocation

    def run(self):
        response = requests.get(
            self.link,
            stream=True,
        )
        printAndLog("Downloading: " + self.link)
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


class SubprocessThread(QThread):
    output = QtCore.Signal(str)

    def __init__(self, command):
        super().__init__()
        self.command = command

    def run(self):
        process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        for line in iter(process.stdout.readline, ""):
            self.output.emit(line.strip())
            printAndLog(line.strip())

        process.stdout.close()
        return_code = process.wait()
        self.output.emit(f"Process finished with return code {return_code}")


# Custom Widgets
class DownloadProgressPopup(QtWidgets.QProgressDialog):
    """
    Runs a download of a file in a thread while reporitng progress to a qt progressbar popup.
    This wont start up in a new process, rather it will just take over the current process
    """

    def __init__(self, link: str, downloadLocation: str, title: str = None):
        super().__init__()
        self.link = link
        self.title = title
        self.downloadLocation = downloadLocation
        self.setup_ui()
        self.startDownload()
        self.exec()
        self.workerThread.wait()

    """
    Initializes all threading bs
    """

    def setup_ui(self):
        self.setLabelText(self.title)
        self.setStyleSheet(styleSheet())
        self.setRange(0, 100)
        self.setMinimumSize(300, 100)
        self.setMaximumSize(300, 100)
        customProgressBar = QtWidgets.QProgressBar()
        customProgressBar.setTextVisible(False)
        self.setAttribute(QtCore.Qt.WA_QuitOnClose)
        customProgressBar.setAlignment(QtCore.Qt.AlignCenter)
        self.setBar(customProgressBar)

    def startDownload(self):
        self.workerThread = DownloadAndReportToQTThread(
            link=self.link, downloadLocation=self.downloadLocation
        )
        self.workerThread.progress.connect(self.setProgress)
        self.workerThread.finished.connect(self.close)
        self.workerThread.finished.connect(self.workerThread.deleteLater)
        self.workerThread.finished.connect(self.workerThread.quit)
        self.workerThread.finished.connect(
            self.workerThread.wait
        )  # need quit and wait to allow process to exit safely
        self.canceled.connect(self.cancel_process)

        self.workerThread.start()

    def cancel_process(self):
        QtWidgets.QApplication.quit()
        sys.exit()

    def setProgress(self, value):
        if self.wasCanceled():
            exit()
        self.setValue(value + 10)


class DisplayCommandOutputPopup(QtWidgets.QDialog):
    """
    Runs a command, and displays the output of said command in the popup
    """

    def __init__(
        self,
        command: str,
        title: str = "Running Command",
        progressBarLength: int = None,
    ):
        super().__init__()
        self.command = command
        self.totalCommandOutput = ""
        self.title = title
        self.totalIters = 0
        self.progressBarLength = progressBarLength
        self.setup_ui()
        self.setLayout(self.gridLayout)
        self.exec()
        self.workerThread.wait()

    """
    Initializes all threading bs
    """

    def setup_ui(self):
        # beginning of bullshit
        self.setWindowTitle(self.title)
        self.setStyleSheet(styleSheet())
        self.setMinimumSize(700, 100)
        self.startDownload()
        self.centralwidget = QtWidgets.QWidget(parent=self)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(parent=self.centralwidget)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.plainTextEdit.setReadOnly(True)
        self.plainTextEdit.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.plainTextEdit.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.gridLayout.addWidget(self.plainTextEdit, 0, 0, 1, 1)
        self.progressBar = QtWidgets.QProgressBar(parent=self.centralwidget)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setTextVisible(False)
        self.progressBar.setObjectName("progressBar")
        if self.progressBarLength:
            self.progressBar.setRange(0, self.progressBarLength)
        self.gridLayout.addWidget(self.progressBar, 1, 0, 1, 1)
        self.widget = QtWidgets.QWidget(parent=self.centralwidget)
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(
            40,
            20,
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        self.horizontalLayout.addItem(spacerItem)
        self.pushButton = QtWidgets.QPushButton(parent=self.widget)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("Cancel")
        self.pushButton.clicked.connect(self.closeEvent)
        self.horizontalLayout.addWidget(self.pushButton)
        self.gridLayout.addWidget(self.widget, 2, 0, 1, 1)

        # end of bullshit

    def closeEvent(self, x):
        self.workerThread.quit()
        self.workerThread.deleteLater()
        self.workerThread.wait()
        self.close()

    def startDownload(self):
        self.workerThread = SubprocessThread(command=self.command)
        self.workerThread.output.connect(self.setProgress)
        self.workerThread.finished.connect(self.close)
        self.workerThread.finished.connect(self.workerThread.deleteLater)
        self.workerThread.finished.connect(self.workerThread.quit)
        self.workerThread.finished.connect(
            self.workerThread.wait
        )  # need quit and wait to allow process to exit safely
        self.workerThread.start()

    def setProgress(self, value):
        self.totalCommandOutput += value + "\n"
        cursor = self.plainTextEdit.textCursor()
        cursor.setVerticalMovementX(-1000000000)
        # updates progressbar based on condition
        if self.progressBarLength is not None:
            if self.totalCommandOutput.count("satisfied") > self.totalIters:
                self.totalIters = self.totalCommandOutput.count("satisfied")
            if self.totalCommandOutput.count("Collecting") > self.totalIters:
                self.totalIters = self.totalCommandOutput.count("Collecting")
            self.progressBar.setValue(self.totalIters)

        self.plainTextEdit.setPlainText(self.totalCommandOutput)
        self.plainTextEdit.setTextCursor(cursor)


if __name__ == "__main__":
    DownloadProgressPopup(
        link="https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/ffmpeg",
        downloadLocation="ffmpeg",
        title="Downloading Python",
    )
