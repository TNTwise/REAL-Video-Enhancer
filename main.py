import sys
import os
import subprocess
from threading import Thread

from PySide6.QtGui import QImage
from PySide6.QtCore import QPropertyAnimation, QPoint, QThread
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog
from mainwindow import Ui_MainWindow  # Import the UI class from the converted module

# other imports
from src.util import (
    checkValidVideo,
    getDefaultOutputVideo,
    getVideoFPS,
    getVideoRes,
    getVideoLength,
    getVideoFrameCount,
)
from src.ProcessTab import ProcessTab


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

        # connect buttons to switch menus
        self.homeBtn.clicked.connect(self.switchToHomePage)
        self.processBtn.clicked.connect(self.switchToProcessingPage)
        self.settingsBtn.clicked.connect(self.switchToSettingsPage)
        self.moreBtn.clicked.connect(self.switchToMorePage)

        # set default home page
        self.stackedWidget.setCurrentIndex(0)

        # connect file select buttons
        self.inputFileSelectButton.clicked.connect(self.openInputFile)
        self.outputFileSelectButton.clicked.connect(self.openOutputFolder)
        # connect render button
        self.startRenderButton.clicked.connect(self.startRender)

    # switch menus
    def switchToHomePage(self):
        self.stackedWidget.setCurrentWidget(self.homePage)  
        self.processBtn.setChecked(False)
        self.settingsBtn.setChecked(False)
        self.moreBtn.setChecked(False)

    def switchToProcessingPage(self):
        self.stackedWidget.setCurrentWidget(self.procPage)
        self.homeBtn.setChecked(False)
        self.settingsBtn.setChecked(False)
        self.moreBtn.setChecked(False)

    def switchToSettingsPage(self):
        self.stackedWidget.setCurrentWidget(self.settingsPage)
        self.homeBtn.setChecked(False)
        self.processBtn.setChecked(False)
        self.moreBtn.setChecked(False)

    def switchToMorePage(self):
        self.stackedWidget.setCurrentWidget(self.morePage)
        self.homeBtn.setChecked(False)
        self.processBtn.setChecked(False)
        self.settingsBtn.setChecked(False)

    def startRender(self):
        processTab = ProcessTab(
            parent=self,
            inputFile=self.inputFile,
            outputPath="",
            videoWidth=self.videoWidth,
            videoHeight=self.videoHeight,
            videoFps=self.videoFps,
            videoFrameCount=self.videoFrameCount,
            upscaleTimes=self.upscaleTimes,
            interpolateTimes=self.interpolateTimes,
        )
        processTab.run()

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
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
