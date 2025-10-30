import sys
import os
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
    Qt,QTimer
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
    QTransform
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
    QCheckBox,
    QComboBox,
    QGridLayout,
    QMainWindow,
)
from multiprocessing import Process
from .QTstyle import styleSheet, Palette
from ..constants import HAS_NETWORK_ON_STARTUP, PLATFORM
from ..Util import log, networkCheck, subprocess_popen_without_terminal

def disable_combobox_item(combobox: QComboBox, index):
    """
    Disable a specific item in a QComboBox (making it visible but unselectable)
    
    Parameters:
        combobox: QComboBox widget
        index: Index of the item to disable
    """
    if 0 <= index < combobox.count():
        model = combobox.model()
        item = model.item(index)
        if item:
            # Remove the enabled flag to disable the item
            flags = item.flags() & ~Qt.ItemIsEnabled
            item.setFlags(flags)
            item.setForeground(QBrush(QColor(128, 128, 128)))  # Gray color

def disable_combobox_item_by_text(combobox: QComboBox, text):
    """
    Disable a specific item in a QComboBox by its text
    
    Parameters:
        combobox: QComboBox widget
        text: Text of the item to disable
    """
    index = combobox.findText(text)
    if index >= 0:
        disable_combobox_item(combobox, index)

def remove_combobox_item_by_text(combobox: QComboBox, text:str,):
    
    """
    removes a specific item in a QComboBox by its text (not case sensitive)
    """
    for idx in range(combobox.count()):
        if text.lower() in combobox.itemText(idx).lower():
            combobox.removeItem(idx)

def hide_layout_widgets(layout):
    # Iterate through all items in the layout and hide the widgets
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item is not None:
            widget = item.widget()
            if widget is not None:
                widget.setVisible(False)  # Hide the widget


def show_layout_widgets(layout):
    # Iterate through all items in the layout and show the widgets
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item is not None:
            widget = item.widget()
            if widget is not None:
                widget.setVisible(True)  # Show the widget

class NotificationOverlay(QWidget):
    def __init__(self, message, parent=None, timeout=3000):
        super().__init__(parent)
        
        # Make this widget a child of the main window, covering its area
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 80);")
        
        # Create layout and label to display the message
        layout = QVBoxLayout(self)
        self.label = QLabel(message, self)
        self.label.setStyleSheet("""
            QLabel {
                color: white;
                padding: 10px;
                font-size: 14px;
                background-color: rgba(50, 50, 50, 180);
                border-radius: 5px;
            }
        """)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label, 0, Qt.AlignCenter)
        
        # Close (hide) this overlay widget after 'timeout' milliseconds
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.close)
        self._timer.start(timeout)
        self.resize(parent.size())
        self.show()

    def resizeEvent(self, event):
        # Match the size of the parent (main window)
        if self.parent():
            self.resize(self.parent().size())
        super().resizeEvent(event)


class UpdateGUIThread(QThread):
    """
    Gets the latest bytes outputed from the shared memory and returns them in QImage format for display
    """

    latestPreviewPixmap = Signal(QtGui.QImage)

    def __init__(self, parent, imagePreviewSharedMemoryID):
        super().__init__()
        self._parent = parent
        self._stop_flag = False  # Boolean flag to control stopping
        self._mutex = QMutex()  # Atomic flag to control stopping
        self.imagePreviewSharedMemoryID = imagePreviewSharedMemoryID
        self.outputVideoHeight = None
        self.outputVideoWidth = None

    def setOutputVideoRes(self, width, height):
        self.outputVideoHeight = height
        self.outputVideoWidth = width

    def run(self):
        while True:
            with QMutexLocker(self._mutex):
                if self._stop_flag:
                    break
            try:
                if self.outputVideoHeight and self.outputVideoWidth:
                    self.shm = shared_memory.SharedMemory(
                        name=self.imagePreviewSharedMemoryID
                    )
                    image_bytes = self.shm.buf[
                        : self.outputVideoHeight * self.outputVideoWidth * 3
                    ].tobytes()
                    expected_size = self.outputVideoHeight * self.outputVideoWidth * 3
                    if len(image_bytes) < expected_size:
                        image_bytes += b"\x00" * (expected_size - len(image_bytes))
                    # Convert image bytes back to numpy array
                    image_array = np.frombuffer(image_bytes, dtype=np.uint8).reshape(
                        (self.outputVideoHeight, self.outputVideoWidth, 3)
                    )
                    pixmap = self.convert_cv_qt(image_array)
                    self.latestPreviewPixmap.emit(pixmap)
            except FileNotFoundError:
                # print("preview not available")
                self.latestPreviewPixmap.emit(None)
            except OSError:
                log("Out of memory.")
            time.sleep(0.2)

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
        log("Downloading: " + self.link)
        if "Content-Length" in response.headers:
            totalByteSize = int(response.headers["Content-Length"])

        else:
            print(
                "Warning: missing key 'Content-Length' in request headers; taking default length of 100 for progress bar."
            )

            totalByteSize = 10000000

        totalSize = 0
        with open(self.downloadLocation, "wb") as f:
            chunk_size = 1024
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                totalSize += chunk_size
                size = totalSize / totalByteSize * 100
                self.progress.emit(size)
        self.finished.emit()


class SubprocessThread(QThread):
    output = QtCore.Signal(str)
    return_code = QtCore.Signal(int)
    fullOutput = QtCore.Signal(str)

    def __init__(self, command):
        super().__init__()
        self.command = command

    def run(self):
        kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "universal_newlines": True,
            "bufsize": 1,
        }

        self.process = subprocess_popen_without_terminal(
            self.command,
            **kwargs,
        )
        totalOutput = ""
        for line in iter(self.process.stdout.readline, ""):
            totalOutput += line
            self.output.emit(line.strip())
            log(line.strip())
        self.fullOutput.emit(totalOutput)
        self.process.stdout.close()
        return_code = self.process.wait()
        self.output.emit(f"Process finished with return code {return_code}")
        self.return_code.emit(return_code)

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
        self.workerThread.return_code.connect(self.setReturnCode)
        self.workerThread.finished.connect(self.close)
        self.workerThread.finished.connect(self.workerThread.deleteLater)
        self.workerThread.finished.connect(self.workerThread.quit)
        self.workerThread.finished.connect(
            self.workerThread.wait
        )  # need quit and wait to allow process to exit safely
        self.workerThread.start()
    
    def setReturnCode(self, return_code):
        self.return_code = return_code

    def setOutput(self, output):
        self.output = output

    def getReturnCode(self):
        return self.return_code

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
        self.startDownload()
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
        self.workerThread.process.terminate()
        self.workerThread.quit()
        self.workerThread.deleteLater()
        self.workerThread.wait()
        self.close()

    def startDownload(self):
        self.workerThread = SubprocessThread(command=self.command)
        self.workerThread.output.connect(self.setProgress)
        self.workerThread.return_code.connect(self.set_return_code)
        self.workerThread.finished.connect(self.close)
        self.workerThread.finished.connect(self.workerThread.deleteLater)
        self.workerThread.finished.connect(self.workerThread.quit)
        self.workerThread.finished.connect(
            self.workerThread.wait
        )  # need quit and wait to allow process to exit safely
        self.workerThread.start()

    def set_return_code(self, return_code):
        self.return_code = return_code
    
    def get_return_code(self) -> int:  
        try:
            return self.return_code
        except Exception:
            return 1

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


class RegularQTPopup(QtWidgets.QDialog):
    def __init__(self, message):
        super().__init__(None)
        self.setWindowTitle("REAL Video Enhancer")
        self.setFixedSize(400, 100)
        self.setStyleSheet(styleSheet())
        
        # Create main layout
        layout = QtWidgets.QVBoxLayout()
        
        # Add message label
        label = QtWidgets.QLabel(message)
        layout.addWidget(label)
        
        # Create horizontal layout for button
        button_layout = QtWidgets.QHBoxLayout()
        
        # Add spacer to push button to the right
        spacer = QtWidgets.QSpacerItem(
            40, 20, 
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum
        )
        button_layout.addItem(spacer)
        
        # Create and add OK button
        ok_button = QtWidgets.QPushButton("OK")
        ok_button.clicked.connect(self.close)
        button_layout.addWidget(ok_button)
        
        # Add button layout to main layout
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.exec()

class IndependentQTPopup(QtWidgets.QDialog):
    def __init__(self, message=None):
        if message is not None:
            self.start(message)
    def start(self, message):
        try:
            app = QApplication(sys.argv)
            app.setStyle("Fusion")
            app.setPalette(Palette())
        except Exception as e:
            log(f"Error starting application: {e}")
            raise e
        super().__init__()
        self.setWindowTitle("REAL Video Enhancer")
        self.setFixedSize(400, 100)
        self.setStyleSheet(styleSheet())
        
        # Create main layout
        layout = QtWidgets.QVBoxLayout()
        
        # Add message label
        label = QtWidgets.QLabel(message)
        layout.addWidget(label)
        
        # Create horizontal layout for button
        button_layout = QtWidgets.QHBoxLayout()
        
        # Add spacer to push button to the right
        spacer = QtWidgets.QSpacerItem(
            40, 20, 
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum
        )
        button_layout.addItem(spacer)
        
        # Add button layout to main layout
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.exec()
        app.exec_()

def NetworkCheckPopup(hostname="https://raw.githubusercontent.com") -> bool:
    if not networkCheck(hostname=hostname):
        RegularQTPopup("No Network Connection")
        return False
    # return true if network connection
    return True

def needs_network_else_exit():
    if not HAS_NETWORK_ON_STARTUP:
        RegularQTPopup("Network is required for this action!\nPlease connect to a network.")
        os._exit(1)


def addNotificationToButton(button: QPushButton):
    notification = QLabel(button)
    notification.setFixedSize(10, 10)
    notification.setStyleSheet("background-color: red; border-radius: 5px;")
    notification.move(button.width() - 15, 5)
    notification.show()


if __name__ == "__main__":
    DownloadProgressPopup(
        link="https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/ffmpeg",
        downloadLocation="ffmpeg",
        title="Downloading Python",
    )
