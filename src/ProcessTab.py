import subprocess
import os
from threading import Thread
import time
import numpy as np
from multiprocessing import shared_memory
import time
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker, Qt
from PySide6 import QtGui
from PySide6.QtGui import QPixmap, QPainter, QPainterPath

from .Util import pythonPath, currentDirectory, modelsPath
from .DownloadModels import DownloadModel


class HandleInputs:
    """A class specifically made to handle the qt widgets and turn their data into a usable command for processing"""

    def __init__(self, parent):
        self.parent = parent


class UpdateGUIThread(QThread):
    """
    Gets the latest bytes outputed from the shared memory and returns them in QImage format for display
    """

    latestPreviewPixmap = Signal(QtGui.QImage)

    def __init__(self, imagePreviewSharedMemoryID, outputVideoHeight, outputVideoWidth):
        super().__init__()
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
                pass
            time.sleep(0.1)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        # rgb_image = cv2.resize(cv_img, (1280, 720)) #Cound resize image if need be
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            cv_img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        return convert_to_Qt_format

    def stop(self):
        with QMutexLocker(self._mutex):
            self._stop_flag = True
        self.shm.close()
        print("Closed Read Memory")


class ProcessTab:
    def __init__(
        self,
        parent,
    ):
        self.parent = parent
        self.imagePreviewSharedMemoryID = "/image_preview"

        self.QConnect()
        self.setupUI()

    def QConnect(self):
        # connect file select buttons
        self.parent.inputFileSelectButton.clicked.connect(self.parent.openInputFile)
        self.parent.outputFileSelectButton.clicked.connect(self.parent.openOutputFolder)
        # connect render button
        self.parent.startRenderButton.clicked.connect(self.parent.startRender)
        self.switchInterpolationAndUpscale()
        self.parent.methodComboBox.currentIndexChanged.connect(
            self.switchInterpolationAndUpscale
        )

    def setupUI(self):
        self.parent.backendComboBox.addItems(self.parent.availableBackends)

    def switchInterpolationAndUpscale(self):
        self.parent.modelComboBox.clear()
        ncnnInterpolateModels = {
            "RIFE 4.6":"rife-v4.6",
            "RIFE 4.15":"rife-v4.15",
            "RIFE 4.18":"rife-v4.18",
            "RIFE 4.20":"rife-v4.20",
        }
        pytorchInterpolateModels = {
            "RIFE 4.6":"rife46.pkl",
            "RIFE 4.15":"rife415.pkl",
            "RIFE 4.18":"rife418.pkl",
            "RIFE 4.20":"rife420.pkl",
        }
        ncnnUpscaleModels = {
            "SPAN (Animation)":"2x_ModenSpanimationV1.5",
        }
        pytorchUpscaleModels = {
            "SPAN (Animation)":"2x_ModenSpanimationV1.5.pth",
        }
        if self.parent.methodComboBox.currentText() == "Interpolate":
            if self.backend == "ncnn":
                models = ncnnInterpolateModels.keys()
            else:
                models = pytorchInterpolateModels.keys()
                
            self.parent.interpolationContainer.setVisible(True)
        if self.parent.methodComboBox.currentText() == "Upscale":
            if self.backend == "ncnn":
                models = ncnnUpscaleModels.keys()
            else:
                models = pytorchUpscaleModels.keys()

            self.parent.interpolationContainer.setVisible(False)
        self.parent.modelComboBox.addItems(models)

    def run(
        self,
        inputFile: str,
        outputPath: str,
        videoWidth: int,
        videoHeight: int,
        videoFps: float,
        videoFrameCount: int,
        upscaleTimes: int,
        interpolateTimes: int,
    ):
        self.inputFile = inputFile
        self.outputPath = outputPath
        self.videoWidth = videoWidth
        self.videoHeight = videoHeight
        self.videoFps = videoFps
        self.videoFrameCount = videoFrameCount
        self.upscaleTimes = upscaleTimes
        self.interpolateTimes = interpolateTimes
        self.outputVideoWidth = videoWidth * upscaleTimes
        self.outputVideoHeight = videoHeight * upscaleTimes
        """
        Function to start the rendering process
        It will initially check for any issues with the current setup, (invalid file, no permissions, etc..)
        Then, based on the settings selected, it will build a command that is then passed into rve-backend
        Finally, It will handle the render via ffmpeg. Taking in the frames from pipe and handing them into ffmpeg on a sperate thread
        """

        self.model = self.parent.modelComboBox.currentText()
        self.backend = self.parent.backendComboBox.currentText()

        # Gui changes
        self.parent.startRenderButton.setEnabled(False)
        DownloadModel(model=self.model, backend=self.backend)
        # self.ffmpegWriteThread()
        writeThread = Thread(target=self.renderToPipeThread)
        writeThread.start()
        self.startGUIUpdate()

    def startGUIUpdate(self):
        self.workerThread = UpdateGUIThread(
            imagePreviewSharedMemoryID=self.imagePreviewSharedMemoryID,
            outputVideoHeight=self.outputVideoHeight,
            outputVideoWidth=self.outputVideoWidth,
        )
        self.workerThread.latestPreviewPixmap.connect(self.updateProcessTab)
        self.workerThread.finished.connect(self.workerThread.deleteLater)
        self.workerThread.finished.connect(self.workerThread.quit)
        self.workerThread.finished.connect(
            self.workerThread.wait
        )  # need quit and wait to allow process to exit safely
        self.workerThread.start()

    def onRenderCompletion(self):
        self.workerThread.stop()
        self.workerThread.quit()
        self.workerThread.wait()
        # reset image preview
        self.parent.previewLabel.clear()

    def renderToPipeThread(self):
        command = [
            f"{pythonPath()}",
            os.path.join(currentDirectory(), "backend", "rve-backend.py"),
            "-i",
            self.inputFile,
            "-o",
            f"{self.outputPath}",
            "--upscaleModel",
            os.path.join(
                modelsPath(), "2x_ModernSpanimationV1.pth"
            ),  # put actual model here, this is a placeholder
            "-b",
            "pytorch",
            "--interpolateFactor",
            "1",
            "--shared_memory_id",
            f"{self.imagePreviewSharedMemoryID}",
            "--half",
        ]

        self.pipeInFrames = subprocess.Popen(
            command,
        )
        self.pipeInFrames.wait()
        print("Done with render")
        self.onRenderCompletion()

    def getRoundedPixmap(self, pixmap, corner_radius):
        size = pixmap.size()
        mask = QPixmap(size)
        mask.fill(Qt.transparent)

        painter = QPainter(mask)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        path = QPainterPath()
        path.addRoundedRect(
            0, 0, size.width(), size.height(), corner_radius, corner_radius
        )

        painter.setClipPath(path)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()

        rounded_pixmap = QPixmap(size)
        rounded_pixmap.fill(Qt.transparent)

        painter = QPainter(rounded_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawPixmap(0, 0, mask)
        painter.end()

        return rounded_pixmap

    def modelNameToFile(self):
        pass

    def updateProcessTab(self, qimage: QtGui.QImage):
        """
        Called by the worker QThread, and updates the GUI elements: Progressbar, Preview, FPS
        """
        try:
            width = self.parent.width()
            height = self.parent.height()
            p = qimage.scaled(width / 2, height / 2, Qt.KeepAspectRatio)
            pixmap = QtGui.QPixmap.fromImage(p)
            roundedPixmap = self.getRoundedPixmap(pixmap, corner_radius=10)
            self.parent.previewLabel.setPixmap(roundedPixmap)
        except FileNotFoundError:
            # print("No preview yet!")
            pass
