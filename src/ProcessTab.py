import subprocess
import os
from threading import Thread
from PySide6.QtCore import QThread
from .Util import ffmpegPath, pythonPath, currentDirectory, modelsPath, activatePythonCommand


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

        self.renderToPipeThread()
        # self.ffmpegWriteThread()
        writeThread = Thread(target=self.ffmpegWriteThread)
        writeThread.start()

    def renderToPipeThread(self):
        command = activatePythonCommand() # have to activate venv
        command += [
            "&&",
            f"python3",
            os.path.join(currentDirectory(), "backend", "rve-backend.py"),
            "-i",
            self.inputFile,
            "-o",
            "PIPE",
            "--interpolateModel",
            os.path.join(modelsPath(),"rife-v4.20-ncnn"),  # put actual model here, this is a placeholder
            "-b",
            "ncnn",
        ]

        self.pipeInFrames = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def ffmpegWriteThread(self):
        command = [
            f"{ffmpegPath()}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{self.videoWidth * self.upscaleTimes}x{self.videoHeight * self.upscaleTimes}",
            "-r",
            f"{self.videoFps * self.interpolateTimes}",
            "-i",
            "-",
            "-i",
            self.inputFile,  # see if audio bugs come up
            "-c:v",
            "libx264",
            f"-crf",
            f"18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            f"{self.outputPath}",
        ]
        writeOutFrames = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            text=True,
            universal_newlines=True,
        )

        outputChunk = (
            self.videoWidth
            * self.videoHeight
            * self.upscaleTimes
            * self.upscaleTimes
            * 3
        )  # 3 is for the channels (RGB)

        totalFrames = int(self.videoFrameCount * self.interpolateTimes)
        for i in range(totalFrames - 1):
            frame = self.pipeInFrames.stdout.read(outputChunk)
            writeOutFrames.stdin.buffer.write(frame)
        writeOutFrames.stdin.close()
        writeOutFrames.wait()

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
