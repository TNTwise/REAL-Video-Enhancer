import sys
import os
import subprocess

from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PySide6.QtCore import Qt
from mainwindow import Ui_MainWindow  # Import the UI class from the converted module

# other imports
from src.Util import (
    checkValidVideo,
    getVideoFPS,
    getVideoRes,
    getVideoLength,
    getVideoFrameCount,
    checkIfDeps,
    pythonPath,
)
from src.ProcessTab import ProcessTab
from src.DownloadTab import DownloadTab
from src.SettingsTab import SettingsTab
from src.DownloadDeps import DownloadDependencies
from src.QTstyle import Palette


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()

        # set up base variables
        self.homeDir = os.path.expanduser("~")
        self.interpolateTimes = 1
        self.upscaleTimes = 2
        self.pipeInFrames = None
        self.latestPreviewImage = None

        # Set up the user interface from Designer.
        self.setupUi(self)
        self.setWindowTitle("REAL Video Enhancer")
        self.setPalette(QApplication.style().standardPalette())
        self.setMinimumSize(1100, 600)
        self.aspect_ratio = self.width() / self.height()
        self.availableBackends = self.getAvailableBackends()

        # set default home page
        self.stackedWidget.setCurrentIndex(0)

        self.QButtonConnect()
        # setup application

        self.setupBackendDeps()

        # set up tabs
        self.processTab = ProcessTab(parent=self)
        self.downloadTab = DownloadTab(parent=self)
        self.settingsTab = SettingsTab(parent=self)
        # self.downloadModels = DownloadModels()

    def QButtonConnect(self):
        # connect buttons to switch menus
        self.homeBtn.clicked.connect(self.switchToHomePage)
        self.processBtn.clicked.connect(self.switchToProcessingPage)
        self.settingsBtn.clicked.connect(self.switchToSettingsPage)
        self.downloadBtn.clicked.connect(self.switchToDownloadPage)

    def setupBackendDeps(self):
        # need pop up window
        if not checkIfDeps():
            downloadDependencies = DownloadDependencies()
            # Dont flip these due to shitty code!
            downloadDependencies.downloadFFMpeg()
            downloadDependencies.downloadPython()

    # switch menus
    def switchToHomePage(self):
        self.stackedWidget.setCurrentWidget(self.homePage)
        self.processBtn.setChecked(False)
        self.settingsBtn.setChecked(False)
        self.downloadBtn.setChecked(False)

    def switchToProcessingPage(self):
        self.stackedWidget.setCurrentWidget(self.procPage)
        self.homeBtn.setChecked(False)
        self.settingsBtn.setChecked(False)
        self.downloadBtn.setChecked(False)

    def switchToSettingsPage(self):
        self.stackedWidget.setCurrentWidget(self.settingsPage)
        self.homeBtn.setChecked(False)
        self.processBtn.setChecked(False)
        self.downloadBtn.setChecked(False)

    def switchToDownloadPage(self):
        self.stackedWidget.setCurrentWidget(self.downloadPage)
        self.homeBtn.setChecked(False)
        self.processBtn.setChecked(False)
        self.settingsBtn.setChecked(False)

    def startRender(self):
        self.processTab.run(
            inputFile=self.inputFileText.text(),
            outputPath=self.outputFileText.text(),
            videoWidth=self.videoWidth,
            videoHeight=self.videoHeight,
            videoFps=self.videoFps,
            videoFrameCount=self.videoFrameCount,
            upscaleTimes=self.upscaleTimes,
            interpolateTimes=self.interpolateTimes,
        )

    def getAvailableBackends(self):
        result = subprocess.run(
            [
                pythonPath(),
                os.path.join("backend", "rve-backend.py"),
                "--list_backends",
            ],
            capture_output=True,
            text=True,
        )

        # Extract the output from the command result
        output = result.stdout.strip()

        # Find the part of the output containing the backends list
        start = output.find("[")
        end = output.find("]") + 1
        backends_str = output[start:end]

        # Convert the string representation of the list to an actual list
        backends = eval(backends_str)

        return backends

    # input file button
    def openInputFile(self):
        """
        Opens a video file and checks if it is valid,

        if it is valid, it will set self.inputFile to the input file, and set the text input field to the input file path.
        if it is not valid, it will give a warning to the user.

        > IMPLEMENT AFTER SELECT AI >  Last, It will enable the output select button, and auto create a default output file

        *NOTE
        This function will set self.videoWidth, self.videoHeight, and self.videoFps

        """

        fileFilter = "Video files (*.mp4 *.mov *.webm)"
        inputFile, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select File",
            dir=self.homeDir,
            filter=fileFilter,
        )

        if checkValidVideo(inputFile):
            self.inputFile = inputFile
            # gets width and height from the res
            self.videoWidth, self.videoHeight = getVideoRes(inputFile)
            # get fps
            self.videoFps = getVideoFPS(inputFile)
            # get video length
            self.videoLength = getVideoLength(inputFile)
            # get video frame count
            self.videoFrameCount = getVideoFrameCount(inputFile)
            self.inputFileText.setText(inputFile)
            self.outputFileText.setEnabled(True)
            self.outputFileSelectButton.setEnabled(True)

    # output file button
    def openOutputFolder(self):
        """
        Opens a folder,
        sets the directory that is selected to the self.outputFolder variable
        sets the outputFileText to the output directory

        It will also read the input file name, and generate an output file based on it.
        """
        self.outputFolder = QFileDialog.getExistingDirectory(
            self,
            caption="Select Output Directory",
            dir=self.homeDir,
        )
        self.outputFileText.setText(self.outputFolder)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # setting the pallette

    app.setPalette(Palette())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
