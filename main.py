import sys
import os
import subprocess
import re

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QGraphicsOpacityEffect,
    QWidget,
)
from PySide6.QtCore import Qt, QPropertyAnimation, QRect, QEasingCurve
from PySide6.QtGui import QIcon
from src.Util import printAndLog
from mainwindow import Ui_MainWindow  # Import the UI class from the converted module
from PySide6 import QtSvg  # Import the QtSvg module so svg icons can be used on windows
from src.version import version

# other imports
from src.Util import (
    checkValidVideo,
    getVideoFPS,
    getVideoRes,
    getVideoLength,
    getVideoFrameCount,
    getVideoEncoder,
    getVideoBitrate,
    checkIfDeps,
    pythonPath,
    openLink,
    getPlatform,
    getOSInfo,
    get_gpu_info,
    getRAMAmount,
    getCPUInfo,
    videosPath,
    
)
from src.ProcessTab import ProcessTab
from src.DownloadTab import DownloadTab
from src.SettingsTab import SettingsTab
from src.MoreTab import MoreTab
from src.DownloadDeps import DownloadDependencies
from src.QTstyle import Palette
from src.QTcustom import DownloadDepsDialog, RegularQTPopup, SettingUpBackendPopup


class MainWindow(QMainWindow, Ui_MainWindow):
    """Main window class for the REAL Video Enhancer application.

    This class extends the QMainWindow and Ui_MainWindow classes to create the main window of the application.
    It sets up the user interface, connects buttons to switch menus, and handles various functionalities such as rendering, file selection, and backend setup.

    Attributes:
        homeDir (str): The home directory path.
        interpolateTimes (int): The number of times to interpolate frames.
        upscaleTimes (int): The number of times to upscale frames.
        pipeInFrames (None): Placeholder for input frames.
        latestPreviewImage (None): Placeholder for the latest preview image.
        aspect_ratio (float): The aspect ratio of the window.

    Methods:
        __init__(): Initializes the MainWindow class.
        QButtonConnect(): Connects buttons to switch menus.
        setupBackendDeps(): Sets up the backend dependencies.
        switchToHomePage(): Switches to the home page.
        switchToProcessingPage(): Switches to the processing page.
        switchToSettingsPage(): Switches to the settings page.
        switchToDownloadPage(): Switches to the download page.
        recursivlyCheckIfDepsOnFirstInstallToMakeSureUserHasInstalledAtLeastOneBackend(): Recursively checks if at least one backend is installed.
        startRender(): Starts the rendering process.
        disableProcessPage(): Disables the process page.
        enableProcessPage(): Enables the process page.
        getAvailableBackends(): Retrieves the available backends.
        openInputFile(): Opens an input video file.
        openOutputFolder(): Opens an output folder.
        killRenderProcess(): Terminates the render process.
        closeEvent(event): Handles the close event of the main window."""

    def __init__(self):
        super().__init__()

        # set up base variables
        self.homeDir = os.path.expanduser("~")
        self.pipeInFrames = None
        self.latestPreviewImage = None
        self.videoWidth=None
        self.videoHeight=None
        self.isVideoLoaded = False

        # setup application
        self.setupBackendDeps()

        # Set up the user interface from Designer.
        self.setupUi(self)
        self.setWindowTitle("REAL Video Enhancer")
        self.setPalette(QApplication.style().standardPalette())
        self.setMinimumSize(1100, 600)

        self.aspect_ratio = self.width() / self.height()

        self.recursivlyCheckIfDepsOnFirstInstallToMakeSureUserHasInstalledAtLeastOneBackend(
            firstIter=True
        )

        # set default home page
        self.stackedWidget.setCurrentIndex(0)

        self.QConnect()
        

        # set up tabs
        self.backendComboBox.addItems(self.availableBackends)
        printOut = (
            "------REAL Video Enhancer------\n"
            + "System Information: \n"
            + "OS: "
            + getOSInfo()
            + "\n"
            + "CPU: "
            + getCPUInfo()
            + "\n"
            + "GPU: "
            + get_gpu_info()
            + "\n"
            + "RAM: "
            + getRAMAmount()
            + "\n"
            + "-------------------------------------------\n"
            + "Software Information: \n"
            + self.fullOutput
        )
        self.renderOutput.setText(printOut)
        printAndLog(printOut)

        halfPrecisionSupport = re.search(
            "half precision support: \s*(true|false)", self.fullOutput.lower()
        )
        if halfPrecisionSupport:
            halfPrecisionSupport = halfPrecisionSupport.group(1) == "true"
        else:
            halfPrecisionSupport = False

        self.processTab = ProcessTab(
            parent=self,
            backend=self.backendComboBox.currentText(),
            method=self.methodComboBox.currentText(),
        )
        
        self.downloadTab = DownloadTab(parent=self)
        self.settingsTab = SettingsTab(
            parent=self, halfPrecisionSupport=halfPrecisionSupport
        )
        self.moreTab = MoreTab(parent=self)
        # Startup Animation
        self.fadeInAnimation(self)

    def setButtonAnimations(self):
        self.homeBtn.enterEvent = self.fade_to_color(self.homeBtn)

    def fadeInAnimation(self, qObject: QWidget, n=None):
        self.opacity_effect = QGraphicsOpacityEffect()
        qObject.setGraphicsEffect(self.opacity_effect)

        self.animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.animation.setDuration(200)  # Duration in milliseconds
        self.animation.setStartValue(0)
        self.animation.setEndValue(1)
        self.animation.start()

    def fadeOutAnimation(self, qObject: QWidget, n=None):
        self.opacity_effect = QGraphicsOpacityEffect()
        qObject.setGraphicsEffect(self.opacity_effect)

        self.animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.animation.setDuration(200)  # Duration in milliseconds
        self.animation.setStartValue(1)
        self.animation.setEndValue(0)
        self.animation.start()

    def fade_to_color(self, color):
        self.animation = QPropertyAnimation(self, b"styleSheet")
        self.animation.setDuration(5000)  # Duration in milliseconds
        self.animation.setStartValue(self.styleSheet())
        self.animation.setEndValue(color)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.animation.start()

    def QConnect(self):
        # connect buttons to switch menus
        self.homeBtn.clicked.connect(self.switchToHomePage)
        self.processBtn.clicked.connect(self.switchToProcessingPage)
        self.settingsBtn.clicked.connect(self.switchToSettingsPage)
        self.downloadBtn.clicked.connect(self.switchToDownloadPage)
        # connect getting default output file
        self.githubBtn.clicked.connect(lambda: openLink("https://github.com/tntwise/REAL-Video-Enhancer"))
        self.kofiBtn.clicked.connect(lambda: openLink("https://ko-fi.com/tntwise"))


    def setupBackendDeps(self):
        # need pop up window
        downloadDependencies = DownloadDependencies()
        downloadDependencies.downloadBackend(version)
        if not checkIfDeps():
            # Dont flip these due to shitty code!
            downloadDependencies.downloadFFMpeg()
            downloadDependencies.downloadPython()
            if getPlatform() == "win32":
                downloadDependencies.downloadVCREDLIST()

    def setButtonsUnchecked(self, buttonToIgnore):
        buttons = [
            self.homeBtn,
            self.processBtn,
            self.settingsBtn,
            self.downloadBtn,
        ]
        for button in buttons:
            if button != buttonToIgnore:
                button.setChecked(False)
            else:
                button.setChecked(True)

    # switch menus
    def switchToHomePage(self):
        self.fadeOutAnimation(self.stackedWidget)
        self.stackedWidget.setCurrentWidget(self.homePage)
        self.setButtonsUnchecked(self.homeBtn)
        self.fadeInAnimation(self.stackedWidget)

    def switchToProcessingPage(self):
        self.fadeOutAnimation(self.stackedWidget)
        self.stackedWidget.setCurrentWidget(self.procPage)
        self.setButtonsUnchecked(self.processBtn)
        self.fadeInAnimation(self.stackedWidget)

    def switchToSettingsPage(self):
        self.fadeOutAnimation(self.stackedWidget)
        self.stackedWidget.setCurrentWidget(self.settingsPage)
        self.setButtonsUnchecked(self.settingsBtn)
        self.fadeInAnimation(self.stackedWidget)

    def switchToDownloadPage(self):
        self.fadeOutAnimation(self.stackedWidget)
        self.stackedWidget.setCurrentWidget(self.downloadPage)
        self.setButtonsUnchecked(self.downloadBtn)
        self.fadeInAnimation(self.stackedWidget)

    def recursivlyCheckIfDepsOnFirstInstallToMakeSureUserHasInstalledAtLeastOneBackend(
        self, firstIter=True
    ):
        """
        will keep trying until the user installs at least 1 backend, happens when user tries to close out of backend slect and gets an error
        """
        try:
            self.availableBackends, self.fullOutput = self.getAvailableBackends()

        except SyntaxError as e:
            printAndLog(str(e))
            if not firstIter:
                RegularQTPopup("Please install at least 1 backend!")
            downloadDependencies = DownloadDependencies()
            DownloadDepsDialog(
                ncnnDownloadBtnFunc=downloadDependencies.downloadNCNNDeps,
                pytorchCUDABtnFunc=downloadDependencies.downloadPyTorchCUDADeps,
                pytorchROCMBtnFunc=downloadDependencies.downloadPyTorchROCmDeps,
                trtBtnFunc=downloadDependencies.downloadTensorRTDeps,
            )
            self.recursivlyCheckIfDepsOnFirstInstallToMakeSureUserHasInstalledAtLeastOneBackend(
                firstIter=False
            )
        
    def generateDefaultOutputFile(self,inputVideo:str, interpolationTimes:int, upscaleTimes:int, videoFps:float, videoWidth:int, videoHeight:int):
        """
        Generates the default output file name based on the input file and the current settings
        """
        file_name = os.path.splitext(os.path.basename(inputVideo))[0]
        self.output_file = f"{file_name}_{interpolationTimes*videoFps}fps_{upscaleTimes*videoWidth}x{upscaleTimes*videoHeight}.mkv"
        iteration=0
        while os.path.isfile(self.output_file):
            self.output_file = f"{file_name}_{interpolationTimes*videoFps}fps_{upscaleTimes*videoWidth}x{upscaleTimes*videoHeight}_({iteration}).mkv"
        return self.output_file
    
    def updateVideoGUIDetails(self):
        if self.isVideoLoaded:
            self.setDefaultOutputFile()
            modelName = self.modelComboBox.currentText()
            method = self.methodComboBox.currentText()
            interpolateTimes = self.getInterpolateTimes(method,modelName)
            scale = self.getScale(method,modelName)
            inputFile=self.inputFileText.text()
            file_extension = os.path.splitext(inputFile)[1]
            text = (
            f"FPS: {round(self.videoFps,0)} -> {round(self.videoFps*interpolateTimes,0)}\n"
            + f"Resolution: {self.videoWidth}x{self.videoHeight} -> {self.videoWidth*scale}x{self.videoHeight*scale}\n"
            + f"Bitrate: {self.videoBitrate}\n"
            + f"Encoder: {self.videoEncoder}\n"
            + f"Container: {file_extension}\n"
            + f"Frame Count: {self.videoFrameCount}\n"
            )
            self.videoInfoTextEdit.setFontPointSize(10)
            self.videoInfoTextEdit.setText(text)
            
    def getScale(self,method,modelName):
        if method == "Upscale":
            scale = (int(re.search(r"\d+x", modelName.lower()).group()[0]))
        elif method == "Interpolate":
            scale = 1
        return scale
    def getInterpolateTimes(self,method,modelName):
        if method == "Upscale":
            interpolateTimes = 1
        elif method == "Interpolate":
            interpolateTimes = int(self.interpolationMultiplierComboBox.currentText())
        return interpolateTimes

    def setDefaultOutputFile(self,useDefaultVideoPath=True):
        """
        Sets the default output file for the video enhancer.
        Parameters:
        - useDefaultVideoPath (bool): Flag indicating whether to use the default video path for the output file.
        Returns:
        None
        """
        
        #check if there is a video loaded
        if self.isVideoLoaded:
            inputFile=self.inputFileText.text()
            modelName = self.modelComboBox.currentText()
            method = self.methodComboBox.currentText()
            interpolateTimes = self.getInterpolateTimes(method,modelName)
            scale = self.getScale(method,modelName)
                
            outputText = self.generateDefaultOutputFile(inputFile, 
                                                        int(interpolateTimes),
                                                        int(scale), round(self.videoFps,0), int(self.videoWidth), int(self.videoHeight))
            if useDefaultVideoPath:
                outputText = os.path.join(videosPath(), outputText)
            self.outputFileText.setText(outputText)
            return outputText
    def startRender(self):
        if self.videoHeight:
            self.startRenderButton.setEnabled(False)
            method = self.methodComboBox.currentText()
            self.progressBar.setRange(
                0,
                # only set the range to multiply the frame count if the method is interpolate
                self.videoFrameCount
                * int(self.interpolationMultiplierComboBox.currentText())
                if method == "Interpolate"
                else self.videoFrameCount,
            )
            self.disableProcessPage()
            self.processTab.run(
                inputFile=self.inputFileText.text(),
                outputPath=self.outputFileText.text(),
                videoWidth=self.videoWidth,
                videoHeight=self.videoHeight,
                videoFps=self.videoFps,
                videoFrameCount=self.videoFrameCount,
                method=method,
                backend=self.backendComboBox.currentText(),
                interpolationTimes=int(self.interpolationMultiplierComboBox.currentText()),
                model=self.modelComboBox.currentText(),
            )
        else:
            pass
            RegularQTPopup("Please select a video file!")

    def disableProcessPage(self):
        self.videoInfoContainer.setDisabled(True)

    def enableProcessPage(self):
        self.videoInfoContainer.setEnabled(True)

    def getAvailableBackends(self):
        output = SettingUpBackendPopup(
            [
                pythonPath(),
                os.path.join("backend", "rve-backend.py"),
                "--list_backends",
            ]
        )
        output = output.getOutput()

        # Find the part of the output containing the backends list
        start = output.find("[")
        end = output.find("]") + 1
        backends_str = output[start:end]

        # Convert the string representation of the list to an actual list
        backends = eval(backends_str)

        return backends, output

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
            self.isVideoLoaded = True
            # gets width and height from the res
            self.videoWidth, self.videoHeight = getVideoRes(inputFile)
            # get fps
            self.videoFps = getVideoFPS(inputFile)
            # get video length
            self.videoLength = getVideoLength(inputFile)
            # get video frame count
            self.videoFrameCount = getVideoFrameCount(inputFile)
            # get video encoder
            self.videoEncoder = getVideoEncoder(inputFile)
            # get video bitrate
            self.videoBitrate = getVideoBitrate(inputFile)
            # get video codec
            self.videoCodec = getVideoEncoder(inputFile)
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
        outputFolder = QFileDialog.getExistingDirectory(
            self,
            caption="Select Output Directory",
            dir=self.homeDir,
        )
        self.outputFileText.setText(os.path.join(outputFolder,self.setDefaultOutputFile(False)))

    def killRenderProcess(self):
        try:  # kills  render process if necessary
            self.renderProcess.terminate()
        except AttributeError:
            printAndLog("No render process!")

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "",
            "Are you sure you want to exit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,  # type: ignore
        )
        if reply == QMessageBox.Yes:  # type: ignore
            self.killRenderProcess()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # setting the pallette

    app.setPalette(Palette())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
