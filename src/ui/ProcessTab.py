import subprocess
import os
from threading import Thread
import re

from PySide6 import QtGui
from PySide6.QtGui import QPixmap, QPainter, QPainterPath
from PySide6.QtCore import Qt, QPropertyAnimation
from BuildFFmpegCommand import BuildFFMpegCommand

from .AnimationHandler import AnimationHandler
from .QTcustom import UpdateGUIThread
from ..Util import (
    pythonPath,
    currentDirectory,
    modelsPath,
    printAndLog,
    log,
    backendDirectory,
)
from ..DownloadModels import DownloadModel
from .SettingsTab import Settings
from ..DiscordRPC import start_discordRPC
from ..ModelHandler import (
    ncnnInterpolateModels,
    ncnnUpscaleModels,
    pytorchInterpolateModels,
    pytorchUpscaleModels,
    tensorrtInterpolateModels,
    tensorrtUpscaleModels,
    onnxUpscaleModels,
    onnxInterpolateModels,
)


class ProcessTab:
    def __init__(self, parent, backend: str, method: str):
        self.parent = parent
        self.imagePreviewSharedMemoryID = "/image_preview" + str(os.getpid())
        self.renderTextOutputList = None
        self.currentFrame = 0
        self.animationHandler = AnimationHandler()
        self.tileUpAnimationHandler = AnimationHandler()
        self.tileDownAnimationHandler = AnimationHandler()
        # encoder dict
        # key is the name in RVE gui
        # value is the encoder used
        

        # get default backend
        self.QConnect()
        self.switchInterpolationAndUpscale()

    def getTotalModels(self, method: str, backend: str) -> dict:
        """
        returns
        the current models available given a method (interpolate, upscale) and a backend (ncnn, tensorrt, pytorch)
        """
        printAndLog("Getting total models, method: " + method + " backend: " + backend)
        if method == "Interpolate":
            match backend:
                case "ncnn":
                    models = ncnnInterpolateModels
                case "pytorch":
                    models = pytorchInterpolateModels
                case "tensorrt":
                    models = tensorrtInterpolateModels
                case "directml":
                    models = onnxInterpolateModels
                case _:
                    log("Error: Invalid backend, returning ncnn models")
                    models = ncnnInterpolateModels  # Return ncnn models if it errors out, this should fix macos
            self.parent.interpolationContainer.setVisible(True)
        if method == "Upscale":
            match backend:
                case "ncnn":
                    models = ncnnUpscaleModels
                case "pytorch":
                    models = pytorchUpscaleModels
                case "tensorrt":
                    models = tensorrtUpscaleModels
                case "directml":
                    models = onnxUpscaleModels
                case _:
                    log("Error: Invalid backend, returning ncnn models")
                    ncnnUpscaleModels  # Return ncnn models if it errors out, this should fix macos
        return models

    def onTilingSwitch(self):
        if self.parent.tilingCheckBox.isChecked():
            self.parent.tileSizeContainer.setVisible(True)
            self.tileDownAnimationHandler.dropDownAnimation(
                self.parent.tileSizeContainer
            )
        else:
            self.tileUpAnimationHandler.moveUpAnimation(self.parent.tileSizeContainer)
            self.parent.tileSizeContainer.setVisible(False)

    def QConnect(self):
        # connect file select buttons

        self.parent.inputFileSelectButton.clicked.connect(self.parent.openInputFile)
        self.parent.inputFileText.textChanged.connect(self.parent.openFileFromYoutubeLink)
        self.parent.outputFileSelectButton.clicked.connect(self.parent.openOutputFolder)
        # connect render button
        self.parent.startRenderButton.clicked.connect(self.parent.startRender)
        cbs = (self.parent.methodComboBox,)
        for combobox in cbs:
            combobox.currentIndexChanged.connect(self.switchInterpolationAndUpscale)
        # set tile size visible to false by default
        self.parent.tileSizeContainer.setVisible(False)
        # connect up tilesize container visiable
        self.parent.tilingCheckBox.stateChanged.connect(self.onTilingSwitch)

        self.parent.interpolationMultiplierSpinBox.valueChanged.connect(
            self.parent.updateVideoGUIDetails
        )
        self.parent.modelComboBox.currentIndexChanged.connect(
            self.parent.updateVideoGUIDetails
        )
        #connect up pausing
        self.parent.pauseRenderButton.setVisible(False)
        self.parent.pauseRenderButton.clicked.connect(self.pauseRender)

    def killRenderProcess(self):
        try:  # kills  render process if necessary
            self.renderProcess.terminate()
        except AttributeError:
            printAndLog("No render process!")

    def switchInterpolationAndUpscale(self):
        """
        Called every render, gets the correct model based on the backend and the method.
        """

        self.parent.modelComboBox.clear()
        # overwrite method
        method = self.parent.methodComboBox.currentText()
        backend = self.parent.backendComboBox.currentText()
        models = self.getTotalModels(method=method, backend=backend)

        self.parent.modelComboBox.addItems(models)
        total_items = self.parent.modelComboBox.count()
        if total_items > 0 and method.lower() == "interpolate":
            self.parent.modelComboBox.setCurrentIndex(total_items - 1)

        if method.lower() == "interpolate":
            self.parent.interpolationContainer.setVisible(True)
            self.parent.upscaleContainer.setVisible(False)
            self.animationHandler.dropDownAnimation(self.parent.interpolationContainer)
        else:
            self.parent.interpolationContainer.setVisible(False)
            self.parent.upscaleContainer.setVisible(True)
            self.animationHandler.dropDownAnimation(self.parent.upscaleContainer)

        self.parent.updateVideoGUIDetails()

    def run(
        self,
        inputFile: str,
        outputPath: str,
        videoWidth: int,
        videoHeight: int,
        videoFps: float,
        videoFrameCount: int,
        tilesize: int,
        tilingEnabled: bool,
        method: str,
        backend: str,
        interpolationTimes: int,
        model: str,
        benchmarkMode: bool,
    ):
        self.inputFile = inputFile
        self.outputPath = outputPath
        self.videoWidth = videoWidth
        self.videoHeight = videoHeight
        self.videoFps = videoFps
        self.tilingEnabled = tilingEnabled
        self.tilesize = tilesize
        self.videoFrameCount = videoFrameCount
        models = self.getTotalModels(method=method, backend=backend)

        # if upscale or interpolate
        """
        Function to start the rendering process
        It will initially check for any issues with the current setup, (invalid file, no permissions, etc..)
        Then, based on the settings selected, it will build a command that is then passed into rve-backend
        Finally, It will handle the render via ffmpeg. Taking in the frames from pipe and handing them into ffmpeg on a sperate thread
        """
        self.benchmarkMode = benchmarkMode
        # get model attributes
        self.modelFile = models[model][0]
        self.downloadFile = models[model][1]
        self.upscaleTimes = models[model][2]
        self.modelArch = models[model][3]

        # get video attributes
        self.outputVideoWidth = videoWidth * self.upscaleTimes
        self.outputVideoHeight = videoHeight * self.upscaleTimes

        # set up pausing
        self.pausedFile = os.path.join(currentDirectory(), os.path.basename(inputFile)+ "_pausedState.txt")
        self.parent.pauseRenderButton.setVisible(True) # switch to pause button on render
        self.parent.startRenderButton.setVisible(False)
        self.parent.startRenderButton.clicked.disconnect()
        self.parent.startRenderButton.clicked.connect(self.resumeRender)


        # get most recent settings
        settings = Settings()
        settings.readSettings()
        self.settings = settings.settings

        # get built ffmpeg command
        buildFFMpegCommand = BuildFFMpegCommand(encoder=self.settings['encoder'],quality=self.settings['video_quality'])
        self.buildFFMpegsettings = buildFFMpegCommand.buildFFmpeg()


        # discord rpc
        if self.settings["discord_rich_presence"] == "True":
            start_discordRPC(method, os.path.basename(self.inputFile), backend)

        DownloadModel(
            modelFile=self.modelFile,
            downloadModelFile=self.downloadFile,
            backend=backend,
        )
        # self.ffmpegWriteThread()

        writeThread = Thread(
            target=lambda: self.renderToPipeThread(
                method=method, backend=backend, interpolateTimes=interpolationTimes
            )
        )
        writeThread.start()
        self.startGUIUpdate()

    def pauseRender(self):
        with open(self.pausedFile,'w') as f:
            f.write("True")
        self.parent.pauseRenderButton.setVisible(False)
        self.parent.startRenderButton.setVisible(True)
        self.parent.startRenderButton.setEnabled(True)
    def resumeRender(self):
        with open(self.pausedFile,'w') as f:
            f.write("False")
        self.parent.pauseRenderButton.setVisible(True)
        self.parent.pauseRenderButton.setEnabled(True)
        self.parent.startRenderButton.setVisible(False)
    def startGUIUpdate(self):
        self.workerThread = UpdateGUIThread(
            parent=self,
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

    def splitListIntoStringWithNewLines(self, string_list: list[str]):
        # Join the strings with newline characters
        return "\n".join(string_list)
        # Set the text to the QTextEdit

    def renderToPipeThread(self, method: str, backend: str, interpolateTimes: int):
        # builds command
        
        command = [
            f"{pythonPath()}",
            "-W",
            "ignore",
            os.path.join(backendDirectory(), "rve-backend.py"),
            "-i",
            self.inputFile,
            "-o",
            f"{self.outputPath}",
            "-b",
            f"{backend}",
            "--precision",
            f"{self.settings['precision']}",
            "--custom_encoder",
            {self.buildFFMpegSettings},
            "--tensorrt_opt_profile",
            f"{self.settings['tensorrt_optimization_level']}",
            "--pausedFile",
            f"{self.pausedFile}"
        ]
        if method == "Upscale":
            command += [
                "--upscaleModel",
                os.path.join(modelsPath(), self.modelFile),
                "--interpolateFactor",
                "1",
            ]
            if self.tilingEnabled:
                command += [
                    "--tilesize",
                    f"{self.tilesize}",
                ]
        if method == "Interpolate":
            command += [
                "--interpolateModel",
                os.path.join(
                    modelsPath(),
                    self.modelFile,
                ),
                "--interpolateFactor",
                f"{interpolateTimes}",
            ]
        if self.settings["preview_enabled"] == "True":
            command += [
                "--shared_memory_id",
                f"{self.imagePreviewSharedMemoryID}",
            ]
        if self.settings["scene_change_detection_enabled"] == "False":
            command += ["--sceneDetectMethod", "none"]
        else:
            command += [
                "--sceneDetectSensitivity",
                self.settings["scene_detection_threshold"],
            ]
        if self.benchmarkMode:
            command += ["--benchmark"]
        self.renderProcess = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        textOutput = []
        for line in iter(self.renderProcess.stdout.readline, b""):
            if self.renderProcess.poll() is not None:
                break  # Exit the loop if the process has terminated
            line = str(line.strip())
            if "it/s" in line:
                textOutput = textOutput[:-1]
            if "FPS" in line:
                textOutput = textOutput[
                    :-2
                ]  # slice the list to only get the last updated data
                self.currentFrame = int(
                    re.search(r"Current Frame: (\d+)", line).group(1)
                )
            textOutput.append(line)
            # self.setRenderOutputContent(textOutput)
            self.renderTextOutputList = textOutput.copy()
            if "Time to complete render" in line:
                break
        log(str(textOutput))
        self.renderProcess.wait()
        # done with render
        # Have to swap the visibility of these here otherwise crash for some reason
        self.parent.pauseRenderButton.setVisible(False)
        self.parent.startRenderButton.setVisible(True)
        self.parent.startRenderButton.setEnabled(True)
        self.parent.onRenderCompletion()

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

        if self.renderTextOutputList is not None:
            # print(self.renderTextOutputList)
            self.parent.renderOutput.setPlainText(
                self.splitListIntoStringWithNewLines(self.renderTextOutputList)
            )
            scrollbar = self.parent.renderOutput.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            self.parent.progressBar.setValue(self.currentFrame)
        if not qimage.isNull():
            width = self.parent.width()
            height = self.parent.height()

            p = qimage.scaled(width / 2, height / 2, Qt.AspectRatioMode.KeepAspectRatio)  # type: ignore
            pixmap = QtGui.QPixmap.fromImage(p)
            roundedPixmap = self.getRoundedPixmap(pixmap, corner_radius=10)
            self.parent.previewLabel.setPixmap(roundedPixmap)
