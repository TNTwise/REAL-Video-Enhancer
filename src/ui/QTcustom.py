import sys
import subprocess
import requests
import time
import numpy as np
from multiprocessing import shared_memory

from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtGui import QIcon
from PySide6.QtCore import (
    QCoreApplication,
    QDate,
    QDateTime,
    QLocale,
    QMetaObject,
    QObject,
    QPoint,
    QRect,
    QSize,
    QTime,
    QUrl,
    Qt,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QConicalGradient,
    QCursor,
    QFont,
    QFontDatabase,
    QGradient,
    QIcon,
    QImage,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPalette,
    QPixmap,
    QRadialGradient,
    QTransform,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QSpacerItem,
)

from .QTstyle import styleSheet
from ..Util import printAndLog, getPlatform
from ..Backendhandler import BackendHandler



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
                image_bytes = self.shm.buf[
                    : self.outputVideoHeight * self.outputVideoWidth * 3
                ].tobytes()

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
        if "Content-Length" in response.headers:
            totalByteSize = int(response.headers["Content-Length"])

        else:
            print(
                "Warning: missing key 'Content-Length' in request headers; taking default length of 100 for progress bar."
            )

            totalByteSize = 10000000

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
    fullOutput = QtCore.Signal(str)

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
        totalOutput = ""
        for line in iter(process.stdout.readline, ""):
            totalOutput += line
            self.output.emit(line.strip())
            printAndLog(line.strip())
        self.fullOutput.emit(totalOutput)
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
        self.setMinimumSize(300, 200)
        self.setMaximumSize(300, 200)
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
            sys.exit()
        self.setValue(value + 10)


class SettingUpBackendPopup(QtWidgets.QDialog):
    """
    Runs a command, and displays the output of said command in the popup
    """

    def __init__(
        self,
        command: list,
        title: str = "Setting Up Backend",
        progressBarLength: int = None,
    ):
        super().__init__()
        self.command = command
        self.totalCommandOutput = ""
        self.title = title
        self.totalIters = 0
        self.progressBarLength = progressBarLength
        self.output = None
        self.setup_ui()
        self.exec()
        self.workerThread.wait()

    """
    Initializes all threading bs
    """

    def setup_ui(self):
        # beginning of bullshit
        self.setWindowTitle(self.title)
        self.setStyleSheet(styleSheet())
        self.setMinimumSize(350, 400)
        self.setMaximumSize(350, 400)
        self.layout2 = QtWidgets.QVBoxLayout()
        self.label = QtWidgets.QLabel(self.title)
        self.iconLabel = QtWidgets.QLabel()
        logobtn = QPushButton()
        icon = QIcon()
        icon.addFile(":/icons/icons/logo-v2.svg", QSize(), QIcon.Normal, QIcon.Off)
        logobtn.setIcon(icon)
        logobtn.setIconSize(QSize(200, 200))
        pixmap = icon.pixmap(QSize(200, 200))
        self.iconLabel.setPixmap(pixmap)
        self.iconLabel.setAlignment(Qt.AlignCenter)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(QFont("Arial", 20))
        self.layout2.addWidget(self.iconLabel)
        self.layout2.addWidget(self.label)
        self.setLayout(self.layout2)
        self.startThread()

        # end of bullshit

    def closeEvent(self, x):
        self.workerThread.quit()
        self.workerThread.deleteLater()
        self.workerThread.wait()
        self.close()

    def startThread(self):
        self.workerThread = SubprocessThread(command=self.command)
        self.workerThread.fullOutput.connect(self.setOutput)
        self.workerThread.finished.connect(self.close)
        self.workerThread.finished.connect(self.workerThread.deleteLater)
        self.workerThread.finished.connect(self.workerThread.quit)
        self.workerThread.finished.connect(
            self.workerThread.wait
        )  # need quit and wait to allow process to exit safely
        self.workerThread.start()

    def setOutput(self, output):
        self.output = output

    def getOutput(self):
        return self.output


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


class DownloadDepsDialog(QtWidgets.QDialog):
    def __init__(
        self,
        ncnnDownloadBtnFunc=None,
        pytorchCUDABtnFunc=None,
        pytorchROCMBtnFunc=None,
        trtBtnFunc=None,
    ):
        super().__init__()
        self.setupUi(self)
        self.closeEventOrig = self.closeEvent
        self.closeEvent = self.closeEventExit
        # pair btns
        self.downloadNCNNBtn.clicked.connect(ncnnDownloadBtnFunc)
        self.downloadTorchCUDABtn.clicked.connect(pytorchCUDABtnFunc)
        self.downloadTorchROCmBtn.clicked.connect(pytorchROCMBtnFunc)
        self.downloadTensorRTBtn.clicked.connect(trtBtnFunc)
        self.pushButton.clicked.connect(self.doneEvent)
        self.setStyleSheet(styleSheet())
        self.exec()

    def closeEventExit(self, x):
        self.close()
        sys.exit()

    def doneEvent(self):
        self.closeEventOrig(QtGui.QCloseEvent())

    def setupUi(self, Dialog):
        Dialog = self
        self.setWindowTitle("Select Dependencies")
        self.verticalLayout = QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.backendSelectContainer = QWidget(Dialog)
        self.backendSelectContainer.setObjectName("backendSelectContainer")
        self.verticalLayout_11 = QVBoxLayout(self.backendSelectContainer)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.label_4 = QLabel(self.backendSelectContainer)
        self.label_4.setObjectName("label_4")
        font = QFont()
        font.setPointSize(25)
        self.label_4.setFont(font)

        self.verticalLayout_11.addWidget(self.label_4)

        self.pytorchBackendInstallerContainer = QWidget(self.backendSelectContainer)
        self.pytorchBackendInstallerContainer.setObjectName(
            "pytorchBackendInstallerContainer"
        )
        self.horizontalLayout_6 = QHBoxLayout(self.pytorchBackendInstallerContainer)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.downloadTorchCUDABtn = QPushButton(self.pytorchBackendInstallerContainer)
        self.downloadTorchCUDABtn.setObjectName("downloadTorchCUDABtn")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.downloadTorchCUDABtn.sizePolicy().hasHeightForWidth()
        )
        self.downloadTorchCUDABtn.setSizePolicy(sizePolicy)
        self.downloadTorchCUDABtn.setMaximumSize(QSize(50, 16777215))
        icon = QIcon()
        icon.addFile(":/icons/icons/download.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.downloadTorchCUDABtn.setIcon(icon)
        self.downloadTorchCUDABtn.setIconSize(QSize(30, 30))

        self.horizontalLayout_6.addWidget(self.downloadTorchCUDABtn)

        self.label_6 = QLabel(self.pytorchBackendInstallerContainer)
        self.label_6.setObjectName("label_6")

        self.horizontalLayout_6.addWidget(self.label_6)

        self.verticalLayout_11.addWidget(self.pytorchBackendInstallerContainer)

        self.pytorchBackendInstallerContainer_2 = QWidget(self.backendSelectContainer)
        self.pytorchBackendInstallerContainer_2.setObjectName(
            "pytorchBackendInstallerContainer_2"
        )
        self.horizontalLayout_8 = QHBoxLayout(self.pytorchBackendInstallerContainer_2)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.downloadTensorRTBtn = QPushButton(self.pytorchBackendInstallerContainer_2)
        self.downloadTensorRTBtn.setObjectName("downloadTensorRTBtn")
        sizePolicy.setHeightForWidth(
            self.downloadTensorRTBtn.sizePolicy().hasHeightForWidth()
        )
        self.downloadTensorRTBtn.setSizePolicy(sizePolicy)
        self.downloadTensorRTBtn.setMaximumSize(QSize(50, 16777215))
        self.downloadTensorRTBtn.setIcon(icon)
        self.downloadTensorRTBtn.setIconSize(QSize(30, 30))

        self.horizontalLayout_8.addWidget(self.downloadTensorRTBtn)

        self.label_8 = QLabel(self.pytorchBackendInstallerContainer_2)
        self.label_8.setObjectName("label_8")

        self.horizontalLayout_8.addWidget(self.label_8)

        self.verticalLayout_11.addWidget(self.pytorchBackendInstallerContainer_2)

        self.pytorchBackendInstallerContainer_3 = QWidget(self.backendSelectContainer)
        self.pytorchBackendInstallerContainer_3.setObjectName(
            "pytorchBackendInstallerContainer_3"
        )
        self.horizontalLayout_9 = QHBoxLayout(self.pytorchBackendInstallerContainer_3)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.downloadTorchROCmBtn = QPushButton(self.pytorchBackendInstallerContainer_3)
        self.downloadTorchROCmBtn.setObjectName("downloadTorchROCmBtn")
        sizePolicy.setHeightForWidth(
            self.downloadTorchROCmBtn.sizePolicy().hasHeightForWidth()
        )
        self.downloadTorchROCmBtn.setSizePolicy(sizePolicy)
        self.downloadTorchROCmBtn.setMaximumSize(QSize(50, 16777215))
        self.downloadTorchROCmBtn.setIcon(icon)
        self.downloadTorchROCmBtn.setIconSize(QSize(30, 30))

        self.horizontalLayout_9.addWidget(self.downloadTorchROCmBtn)
        
        
        backendHandler = BackendHandler(self)
        backendHandler.enableCorrectBackends()
        self.label_9 = QLabel(self.pytorchBackendInstallerContainer_3)
        self.label_9.setObjectName("label_9")

        self.horizontalLayout_9.addWidget(self.label_9)

        self.verticalLayout_11.addWidget(self.pytorchBackendInstallerContainer_3)

        self.pytorchBackendInstallerContainer_4 = QWidget(self.backendSelectContainer)
        self.pytorchBackendInstallerContainer_4.setObjectName(
            "pytorchBackendInstallerContainer_4"
        )
        self.horizontalLayout_10 = QHBoxLayout(self.pytorchBackendInstallerContainer_4)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.downloadNCNNBtn = QPushButton(self.pytorchBackendInstallerContainer_4)
        self.downloadNCNNBtn.setObjectName("downloadNCNNBtn")
        sizePolicy.setHeightForWidth(
            self.downloadNCNNBtn.sizePolicy().hasHeightForWidth()
        )
        self.downloadNCNNBtn.setSizePolicy(sizePolicy)
        self.downloadNCNNBtn.setMaximumSize(QSize(50, 16777215))
        self.downloadNCNNBtn.setIcon(icon)
        self.downloadNCNNBtn.setIconSize(QSize(30, 30))

        self.horizontalLayout_10.addWidget(self.downloadNCNNBtn)

        self.label_10 = QLabel(self.pytorchBackendInstallerContainer_4)
        self.label_10.setObjectName("label_10")

        self.horizontalLayout_10.addWidget(self.label_10)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.horizontalSpacer = QSpacerItem(
            40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.pushButton = QPushButton(self.backendSelectContainer)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("Done")

        self.horizontalLayout.addWidget(self.pushButton)

        self.verticalLayout_11.addWidget(self.pytorchBackendInstallerContainer_4)

        self.verticalLayout.addWidget(self.backendSelectContainer)

        self.verticalLayout_11.addLayout(self.horizontalLayout)

        self.retranslateUi(Dialog)

        QMetaObject.connectSlotsByName(Dialog)

    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", "Dialog", None))
        self.label_4.setText(QCoreApplication.translate("Dialog", "Backends", None))
        self.downloadTorchCUDABtn.setText("")
        self.label_6.setText(
            QCoreApplication.translate("Dialog", "PyTorch CUDA (Nvidia Only) ", None)
        )
        self.downloadTensorRTBtn.setText("")
        self.label_8.setText(
            QCoreApplication.translate(
                "Dialog", "TensorRT (Nvidia RTX 20 series and up)", None
            )
        )
        self.downloadTorchROCmBtn.setText("")
        self.label_9.setText(
            QCoreApplication.translate(
                "Dialog", "PyTorch ROCm (AMD RX 6000 through RX 7000, linux only)", None
            )
        )
        self.downloadNCNNBtn.setText("")
        self.label_10.setText(
            QCoreApplication.translate("Dialog", "NCNN Vulkan (All GPUs, Slower)", None)
        )

    # retranslateUi


class RegularQTPopup(QtWidgets.QDialog):
    def __init__(self, message):
        super().__init__()
        self.setWindowTitle(message)
        self.setFixedSize(300, 100)
        layout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel(message)
        layout.addWidget(label)
        self.setLayout(layout)
        self.exec()


if __name__ == "__main__":
    DownloadProgressPopup(
        link="https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/ffmpeg",
        downloadLocation="ffmpeg",
        title="Downloading Python",
    )
