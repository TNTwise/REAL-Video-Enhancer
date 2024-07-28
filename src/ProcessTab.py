import subprocess
import os
from threading import Thread
import sys
from PySide6.QtCore import QThread
from .Util import ffmpegPath, pythonPath, currentDirectory, modelsPath


class ProcessTab:
    def __init__(
        self,
        parent,
    ):
        self.parent = parent
        self.QButtonConnect()

    def QButtonConnect(self):
        # connect file select buttons
        self.parent.inputFileSelectButton.clicked.connect(self.parent.openInputFile)
        self.parent.outputFileSelectButton.clicked.connect(self.parent.openOutputFolder)
        # connect render button
        self.parent.startRenderButton.clicked.connect(self.parent.startRender)

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
        """
        Function to start the rendering process
        It will initially check for any issues with the current setup, (invalid file, no permissions, etc..)
        Then, based on the settings selected, it will build a command that is then passed into rve-backend
        Finally, It will handle the render via ffmpeg. Taking in the frames from pipe and handing them into ffmpeg on a sperate thread
        """

        # Gui changes
        self.parent.startRenderButton.setEnabled(False)

        
        # self.ffmpegWriteThread()
        writeThread = Thread(target=self.renderToPipeThread)
        writeThread.start()

    def renderToPipeThread(self):
        command = [
            f"{pythonPath()}",
            os.path.join(currentDirectory(), "backend", "rve-backend.py"),
            "-i",
            self.inputFile,
            "-o",
            f"{self.outputPath}",
            "--interpolateModel",
            os.path.join(modelsPath(),"rife-v4.20-ncnn"),  # put actual model here, this is a placeholder
            "-b",
            "ncnn",
        ]

        self.pipeInFrames = subprocess.run(
            command,
            

        )

    
    def updateProcessTab(self):
        """
        Called by the worker QThread, and updates the GUI elements: Progressbar, Preview, FPS
        """
        pass
        """if self.latestPreviewImage is not None:
            height, width, channel = self.latestPreviewImage.shape
            bytesPerLine = 3 * width
            qImg = QImage(self.latestPreviewImage.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.previewLabel.setPixmap(qImg)"""
