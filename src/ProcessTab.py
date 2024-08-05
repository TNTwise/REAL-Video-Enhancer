from math import e
import subprocess
import os
from threading import Thread
import time
import numpy as np
from multiprocessing import shared_memory
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker, Qt
from PySide6 import QtGui
from PySide6.QtGui import QPixmap, QPainter, QPainterPath

from backend.src.Util import printAndLog

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


class ProcessTab:
    def __init__(
        self,
        parent,
    ):
        self.parent = parent
        self.imagePreviewSharedMemoryID = "/image_preview"
        # get default backend
        self.setupUI()
        self.QConnect()

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
        self.backend = self.parent.backendComboBox.currentText()
        """
        Key value pairs of the model name in the GUI
        Data inside the tuple:
        [0] = file in models directory
        [1] = file to download
        [2] = upscale times
        [3] = arch
        """
        self.ncnnInterpolateModels = {
            "RIFE 4.6": ("rife-v4.6", "rife-v4.6.tar.gz", 1, "rife46"),
            "RIFE 4.7": ("rife-v4.7", "rife-v4.7.tar.gz", 1, "rife47"),
            "RIFE 4.15": ("rife-v4.15", "rife-v4.15.tar.gz", 1, "rife413"),
            "RIFE 4.18": ("rife-v4.18", "rife-v4.18.tar.gz", 1, "rife413"),
            "RIFE 4.20": ("rife-v4.20", "rife-v4.20.tar.gz", 1, "rife420"),
            "RIFE 4.21": ("rife-v4.21", "rife-v4.21.tar.gz", 1, "rife421"),
        }
        self.pytorchInterpolateModels = {
            "RIFE 4.6": ("rife4.6.pkl", "rife4.6.pkl", 1, "rife46"),
            "RIFE 4.7": ("rife4.7.pkl", "rife4.7.pkl", 1, "rife47"),
            "RIFE 4.15": ("rife4.15.pkl", "rife4.15.pkl", 1, "rife413"),
            "RIFE 4.18": ("rife4.18.pkl", "rife4.18.pkl", 1, "rife413"),
            "RIFE 4.20": ("rife4.20.pkl", "rife4.20.pkl", 1, "rife420"),
            "RIFE 4.21": ("rife4.21.pkl", "rife4.21.pkl", 1, "rife421"),
        }
        self.ncnnUpscaleModels = {
            "SPAN (Animation) (2X)": (
                "2x_ModenSpanimationV1.5",
                "2x_ModenSpanimationV1.5.tar.gz",
                2,
                "SPAN",
            ),
        }
        self.pytorchUpscaleModels = {
            "SPAN (Animation) (2X)": (
                "2x_ModenSpanimationV1.5.pth",
                "2x_ModenSpanimationV1.5.pth",
                2,
                "SPAN",
            ),
        }

        models = None
        if self.parent.methodComboBox.currentText() == "Interpolate":
            if self.backend == "ncnn":
                models = self.ncnnInterpolateModels.keys()
                self.totalModels = self.ncnnInterpolateModels | self.ncnnUpscaleModels
            else:
                models = self.pytorchInterpolateModels.keys()
                self.totalModels = (
                    self.pytorchInterpolateModels | self.pytorchUpscaleModels
                )

            self.parent.interpolationContainer.setVisible(True)
        if self.parent.methodComboBox.currentText() == "Upscale":
            if self.backend == "ncnn":
                models = self.ncnnUpscaleModels.keys()
                self.totalModels = self.ncnnInterpolateModels | self.ncnnUpscaleModels
            else:
                models = self.pytorchUpscaleModels.keys()
                self.totalModels = (
                    self.pytorchInterpolateModels | self.pytorchUpscaleModels
                )
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
        method: str,
    ):
        self.inputFile = inputFile
        self.outputPath = outputPath
        self.videoWidth = videoWidth
        self.videoHeight = videoHeight
        self.videoFps = videoFps
        self.videoFrameCount = videoFrameCount
        self.interpolateTimes = int(
            self.parent.interpolationMultiplierComboBox.currentText()
        )
        self.model = self.parent.modelComboBox.currentText()
        # get model attributes
        self.upscaleTimes = self.totalModels[self.model][2]
        self.modelArch = self.totalModels[self.model][3]
        # get video attributes
        self.outputVideoWidth = videoWidth * self.upscaleTimes
        self.outputVideoHeight = videoHeight * self.upscaleTimes
        # if upscale or interpolate
        self.method = method
        """
        Function to start the rendering process
        It will initially check for any issues with the current setup, (invalid file, no permissions, etc..)
        Then, based on the settings selected, it will build a command that is then passed into rve-backend
        Finally, It will handle the render via ffmpeg. Taking in the frames from pipe and handing them into ffmpeg on a sperate thread
        """

        self.backend = self.parent.backendComboBox.currentText()

        # Gui changes
        self.parent.startRenderButton.setEnabled(False)
        self.modelFile = self.totalModels[self.model][0]
        self.downloadFile = self.totalModels[self.model][1]
        DownloadModel(
            modelFile=self.modelFile,
            downloadModelFile=self.downloadFile,
            backend=self.backend,
        )
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
        self.parent.startRenderButton.setEnabled(True)

    def renderToPipeThread(self):
        # builds command
        command = [
            f"{pythonPath()}",
            os.path.join(currentDirectory(), "backend", "rve-backend.py"),
            "-i",
            self.inputFile,
            "-o",
            f"{self.outputPath}",
            "-b",
            f"{self.backend}",
            "--shared_memory_id",
            f"{self.imagePreviewSharedMemoryID}",
            "--half",
        ]
        if self.method == "Upscale":
            command += [
                "--upscaleModel",
                os.path.join(modelsPath(), self.modelFile),
            ]
        if self.method == "Interpolate":
            command += [
                "--interpolateModel",
                os.path.join(
                    modelsPath(),
                    self.modelFile,
                ),
                "--interpolateArch",
                f"{self.modelArch}",
                "--interpolateFactor",
                f"{self.interpolateTimes}",
            ]
        self.parent.renderProcess = subprocess.Popen(
            command,
        )
        self.parent.renderProcess.wait()
        # done with render
        self.onRenderCompletion()

    def getRoundedPixmap(self, pixmap, corner_radius):
        size = pixmap.size()
        mask = QPixmap(size)
        mask.fill(Qt.transparent)  # type: ignore

        painter = QPainter(mask)
        painter.setRenderHint(QPainter.Antialiasing)  # type: ignore
        painter.setRenderHint(QPainter.SmoothPixmapTransform)  # type: ignore

        path = QPainterPath()
        path.addRoundedRect(
            0, 0, size.width(), size.height(), corner_radius, corner_radius
        )

        painter.setClipPath(path)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()

        rounded_pixmap = QPixmap(size)
        rounded_pixmap.fill(Qt.transparent)  # type: ignore

        painter = QPainter(rounded_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)  # type: ignore
        painter.setRenderHint(QPainter.SmoothPixmapTransform)  # type: ignore
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

            p = qimage.scaled(width / 2, height / 2, Qt.AspectRatioMode.KeepAspectRatio)  # type: ignore
            pixmap = QtGui.QPixmap.fromImage(p)
            roundedPixmap = self.getRoundedPixmap(pixmap, corner_radius=10)
            self.parent.previewLabel.setPixmap(roundedPixmap)
        except FileNotFoundError:
            # print("No preview yet!")
            pass
