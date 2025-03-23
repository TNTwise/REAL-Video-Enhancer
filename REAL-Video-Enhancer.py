import sys
import os

# patch for macos
if sys.platform == "darwin":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # this goes one step up, and goes into the actual directory. This is where backend will be copied to.
    os.chdir("..")
import math
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QMessageBox,
)
from PySide6.QtGui import QIcon
from mainwindow import Ui_MainWindow
from PySide6 import QtSvg  # Import the QtSvg module so svg icons can be used on windows
from src.version import version
from src.InputHandler import VideoLoader
from src.ModelHandler import getCustomModelScale, getModels

# other imports
from src.Util import (
    getOSInfo,
    getRAMAmount,
    getCPUInfo,
    checkForWritePermissions,
    getAvailableDiskSpace,
    FileHandler,
    log,
)
from src.DownloadModels import DownloadModel
from src.constants import CUSTOM_MODELS_PATH, MODELS_PATH
from src.ui.ProcessTab import ProcessTab
from src.ui.DownloadTab import DownloadTab
from src.ui.SettingsTab import SettingsTab, Settings
from src.ui.HomeTab import HomeTab
from src.Backendhandler import BackendHandler
from src.ModelHandler import totalModels
from src.ui.AnimationHandler import AnimationHandler
from src.ui.QTstyle import Palette
from src.ui.QTcustom import RegularQTPopup, NotificationOverlay
from src.ui.RenderQueue import RenderQueue, RenderOptions

svg = (
    QtSvg.QSvgRenderer()
)  # utilize the imported QtSvg module to render svg icons on windows


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
        self.videoWidth = None
        self.videoHeight = None
        self.isVideoLoaded = False

        # setup application

        # Set up the user interface from Designer.
        self.setupUi(self)
        backendHandler = BackendHandler(self)
        backendHandler.enableCorrectBackends()

        self.renderQueue = RenderQueue(self.renderQueueListWidget)

        backendHandler.setupBackendDeps()
        self.backends, self.fullOutput = (
            backendHandler.recursivlyCheckIfDepsOnFirstInstallToMakeSureUserHasInstalledAtLeastOneBackend(
                firstIter=True
            )
        )

        backendHandler.hideUninstallButtons()
        backendHandler.showUninstallButton(self.backends)
        icon_path = ":/icons/icons/logo-v2.svg"
        self.setWindowIcon(QIcon(icon_path))
        QApplication.setWindowIcon(QIcon(icon_path))
        self.setWindowTitle("REAL Video Enhancer")
        self.setPalette(QApplication.style().standardPalette())
        self.setMinimumSize(1100, 700)

        self.aspect_ratio = self.width() / self.height()

        # set default home page
        self.stackedWidget.setCurrentIndex(0)

        self.QConnect()
        # set up tabs
        self.backendComboBox.addItems(self.backends)
        printOut = (
            "System Information:\n\n"
            + "OS: "
            + getOSInfo()
            + "\n"
            + "CPU: "
            + getCPUInfo()
            + "\n"
            + "RAM: "
            + getRAMAmount()
            + "\n"
            + "Available Disk Space: "
            + str(round(getAvailableDiskSpace(), 2))
            + "GB"
            + "\n"
            + "\nSoftware Information:\n\n"
            + f"REAL Video Enhancer Version: {version}\n"
            + self.fullOutput
        )
        self.systemInfoText.setText(printOut)
        log(printOut)

        # process the output
        total_ncnn_gpus = -1
        total_pytorch_gpus = -1
        for line in self.fullOutput.lower().split("\n"):
            if "half precision support:" in line:
                halfPrecisionSupport = "true" in line
            if "ncnn gpu " in line:  # this is to grab every line with "GPU "
                total_ncnn_gpus += 1
            if "pytorch gpu " in line:
                total_pytorch_gpus += 1

        total_pytorch_gpus = max(0, total_pytorch_gpus)  # minimum gpu id is 0
        total_ncnn_gpus = max(0, total_ncnn_gpus)

        settings = Settings()
        settings.readSettings()
        self.settings = settings
        self.processTab = ProcessTab(
            parent=self,
            settings=settings,
        )
        self.homeTab = HomeTab(parent=self)
        self.downloadTab = DownloadTab(parent=self, backends=self.backends)
        self.settingsTab = SettingsTab(
            parent=self,
            halfPrecisionSupport=halfPrecisionSupport,
            total_ncnn_gpus=total_ncnn_gpus,
            total_pytorch_gpus=total_pytorch_gpus,
        )

        # Startup Animation
        self.animationHandler = AnimationHandler()
        self.animationHandler.fadeInAnimation(self)

    def QConnect(self):
        # connect buttons to switch menus
        self.homeBtn.clicked.connect(self.switchToHomePage)
        self.processBtn.clicked.connect(self.switchToProcessingPage)
        self.settingsBtn.clicked.connect(self.switchToSettingsPage)
        self.downloadBtn.clicked.connect(self.switchToDownloadPage)
        # connect getting default output file

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
        self.animationHandler.fadeOutAnimation(self.stackedWidget)
        self.stackedWidget.setCurrentWidget(self.homePage)
        self.setButtonsUnchecked(self.homeBtn)
        self.animationHandler.fadeInAnimation(self.stackedWidget)

    def switchToProcessingPage(self):
        self.animationHandler.fadeOutAnimation(self.stackedWidget)
        self.stackedWidget.setCurrentWidget(self.procPage)
        self.setButtonsUnchecked(self.processBtn)
        self.animationHandler.fadeInAnimation(self.stackedWidget)

    def switchToSettingsPage(self):
        self.animationHandler.fadeOutAnimation(self.stackedWidget)
        self.stackedWidget.setCurrentWidget(self.settingsPage)
        self.setButtonsUnchecked(self.settingsBtn)
        self.animationHandler.fadeInAnimation(self.stackedWidget)

    def switchToDownloadPage(self):
        self.animationHandler.fadeOutAnimation(self.stackedWidget)
        self.stackedWidget.setCurrentWidget(self.downloadPage)
        self.setButtonsUnchecked(self.downloadBtn)
        self.animationHandler.fadeInAnimation(self.stackedWidget)

    def updateVideoGUIText(self):
        if self.isVideoLoaded:
            upscaleModelName = self.upscaleModelComboBox.currentText()
            interpolateModelName = self.interpolateModelComboBox.currentText()
            interpolateTimes = self.getInterpolationMultiplier(interpolateModelName)
            scale = self.getUpscaleModelScale(upscaleModelName)
            text = (
                f"FPS: {round(self.videoFps, 0)} -> {round(self.videoFps * interpolateTimes, 0)}\n"
                + f"Resolution: {self.videoWidth}x{self.videoHeight} -> {self.videoWidth * scale}x{self.videoHeight * scale}\n"
                + f"Frame Count: {self.videoFrameCount} -> {int(round(self.videoFrameCount * interpolateTimes, 0))}\n"
                + f"Bitrate: {self.videoBitrate}\n"
                + f"Encoder: {self.videoEncoder}\n"
                + f"Container: {self.videoContainer}\n"
            )
            self.videoInfoTextEdit.setFontPointSize(10)
            self.videoInfoTextEdit.setText(text)

    def getInterpolationMultiplier(self, interpolateModelName):
        if interpolateModelName == "None":
            interpolateTimes = 1
        else:
            interpolateTimes = self.interpolationMultiplierSpinBox.value()
        return interpolateTimes

    def getUpscaleModelScale(self, upscaleModelName):
        if upscaleModelName == "None" or upscaleModelName == "":
            scale = 1
        else:
            scale = totalModels[upscaleModelName][2]
        return scale

    def setDefaultOutputFile(self, outputDirectory):
        """
        Sets the default output file for the video enhancer.
        Parameters:
        - useDefaultVideoPath (bool): Flag indicating whether to use the default video path for the output file.
        Returns:
        None
        """

        # check if there is a video loaded
        if self.isVideoLoaded:
            inputFile = self.inputFileText.text()
            upscaleModelName = self.upscaleModelComboBox.currentText()
            interpolateModelName = self.interpolateModelComboBox.currentText()
            interpolateTimes = self.getInterpolationMultiplier(interpolateModelName)
            scale = self.getUpscaleModelScale(upscaleModelName)
            container = self.settings.settings["video_container"]

            file_name = os.path.splitext(os.path.basename(inputFile))[0]
            base_file_name = (
                f"{file_name}"
                + f"_{round(interpolateTimes * self.videoFps, 0)}fps"
                + f"_{scale * self.videoWidth}x{scale * self.videoHeight}"
            )
            output_file = os.path.join(
                outputDirectory,
                f"{base_file_name}.{container}",
            )
            iteration = 0
            while os.path.isfile(output_file):
                output_file = os.path.join(
                    outputDirectory,
                    f"{base_file_name}_({iteration}).{container}",
                )
                iteration += 1
            self.outputFileText.setText(output_file)
            return output_file

    def updateVideoGUIDetails(self):
        isInterpolate = self.interpolateModelComboBox.currentText() != "None"
        isUpscale = self.upscaleModelComboBox.currentText() != "None"

        self.interpolationContainer.setVisible(isInterpolate)
        self.interpolateContainer_2.setVisible(isInterpolate)
        # set interpolation container visible if interpolate model is not none
        self.upscaleContainer.setVisible(isUpscale)
        self.settings.readSettings()
        self.setDefaultOutputFile(self.settings.settings["output_folder_location"])
        self.updateVideoGUIText()

    def addToRenderQueue(self):
        self.settings.readSettings()
        interpolate = self.interpolateModelComboBox.currentText()
        upscale = self.upscaleModelComboBox.currentText()
        output_path = self.outputFileText.text()
        if interpolate == "None":
            interpolate = None
        if upscale == "None":
            upscale = None

        if not self.isVideoLoaded:
            NotificationOverlay("Video is not loaded!", self, timeout=1500)
            return

        if not interpolate and not upscale:
            NotificationOverlay("Please select at least one model!", self, timeout=1500)
            return

        for renderOptions in self.renderQueue.getQueue():
            if output_path == renderOptions.outputPath:
                NotificationOverlay("Output file already in queue!", self, timeout=1500)
                return

        backend = self.backendComboBox.currentText()
        upscaleTimes = 1
        upscaleModelArch = "custom"
        interpolateModels, upscaleModels = getModels(backend)

        if interpolate:
            interpolateDownloadFile = interpolateModels[interpolate][1]
            interpolateModelFile = interpolateModels[interpolate][0]

            dm = DownloadModel(
                modelFile=interpolateModelFile,
                downloadModelFile=interpolateDownloadFile,
            )
            if not dm.downloadModel():
                NotificationOverlay(
                    "Unable to download model, please check your network and try again.",
                    self,
                    timeout=1500,
                )
                return

        if upscale:
            upscaleModelFile = upscaleModels[upscale][0]
            upscaleDownloadFile = upscaleModels[upscale][1]
            upscaleTimes = upscaleModels[upscale][2]
            upscaleModelArch = upscaleModels[upscale][3]
            if upscaleModelArch != "custom":
                dm = DownloadModel(
                    modelFile=upscaleModelFile,
                    downloadModelFile=upscaleDownloadFile,
                )
                if not dm.downloadModel():
                    NotificationOverlay(
                        "Unable to add to render queue.\nModel can't be downloaded.\nPlease check your network and try again.",
                        self,
                        timeout=2500,
                    )
                    return

        renderOptions = RenderOptions(
            inputFile=self.inputFileText.text(),
            outputPath=output_path,
            videoWidth=self.videoWidth,
            videoHeight=self.videoHeight,
            videoFps=self.videoFps,
            tilingEnabled=self.tilingCheckBox.isChecked(),
            tilesize=self.tileSizeComboBox.currentText(),
            videoFrameCount=self.videoFrameCount,
            backend=self.backendComboBox.currentText(),
            interpolateModel=interpolate,
            upscaleModel=upscale,
            interpolateTimes=self.getInterpolationMultiplier(
                self.interpolateModelComboBox.currentText()
            ),
            benchmarkMode=self.benchmarkModeCheckBox.isChecked(),
            sloMoMode=self.sloMoModeCheckBox.isChecked(),
            dyanmicScaleOpticalFlow=self.dynamicScaledOpticalFlowCheckBox.isChecked(),
            ensemble=self.ensembleCheckBox.isChecked(),
            upscaleModelArch=upscaleModelArch,
            upscaleTimes=upscaleTimes,
            upscaleModelFile=upscaleModelFile if upscale else None,
            interpolateModelFile=interpolateModelFile if interpolate else None,
            hdrMode=self.hdrModeCheckBox.isChecked(),
        )
        # alert user that item has been added to queue
        NotificationOverlay(
            f"{self.inputFileText.text()} Added to queue!", self, timeout=1500
        )
        self.renderQueue.add(renderOptions)

    def startRender(self):
        if len(self.renderQueue.queue) == 0:
            NotificationOverlay("Render queue is empty!", self, timeout=1500)
            return
        self.startRenderButton.setEnabled(False)

        self.disableProcessPage()
        self.processTab.run(self.renderQueue)

    def disableProcessPage(self):
        for child in self.generalSettings.children():
            child.setEnabled(False)
        for child in self.advancedSettings.children():
            child.setEnabled(False)
        for child in self.renderQueueTab.children():
            child.setEnabled(False)

    def enableProcessPage(self):
        for child in self.generalSettings.children():
            child.setEnabled(True)
        for child in self.advancedSettings.children():
            child.setEnabled(True)
        for child in self.renderQueueTab.children():
            child.setEnabled(True)

    def loadVideo(self, inputFile):
        videoHandler = VideoLoader(inputFile)
        videoHandler.loadVideo()
        if (
            not videoHandler.isValidVideo()
        ):  # this handles case for invalid youtube link and invalid video file
            NotificationOverlay("Not a valid input!", self, timeout=1500)
            return
        videoHandler.getData()
        self.videoWidth = videoHandler.width
        self.videoHeight = videoHandler.height
        self.videoFps = videoHandler.fps
        self.videoLength = videoHandler.duration
        self.videoFrameCount = videoHandler.total_frames
        self.videoEncoder = videoHandler.codec_str
        self.videoBitrate = videoHandler.bitrate
        self.videoContainer = videoHandler.videoContainer

        self.inputFileText.setText(inputFile)
        self.outputFileText.setEnabled(True)
        self.outputFileSelectButton.setEnabled(True)
        self.isVideoLoaded = True
        self.updateVideoGUIDetails()

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

        fileFilter = "Video files (*.mp4 *.mov *.webm *.mkv);;All files (*.*)"
        inputFile, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select File",
            dir=self.homeDir,
            filter=fileFilter,
        )
        self.loadVideo(inputFile)

    def importCustomModel(self, format: str):
        """
        *args
        format: str
            The format of the model to import (pytorch, ncnn)
        """

        if format == "pytorch":
            fileFilter = "PyTorch Model (*.pth *.safetensors)"

            modelFile, _ = QFileDialog.getOpenFileName(
                parent=self,
                caption="Select PyTorch Model",
                dir=self.homeDir,
                filter=fileFilter,
            )
            if not getCustomModelScale(os.path.basename(modelFile)):
                NotificationOverlay(
                    "Custom model does not have a valid\nupscale factor in the name.\nExample: 2x or x2. Skipping import...",
                    self,
                    timeout=1500,
                )
                return

            outputModelPath = os.path.join(
                CUSTOM_MODELS_PATH, os.path.basename(modelFile)
            )
            FileHandler.copyFile(modelFile, CUSTOM_MODELS_PATH)

            if os.path.isfile(outputModelPath):
                NotificationOverlay(
                    "Model imported successfully!\nPlease restart the app for the changes to take effect.",
                    self,
                    timeout=1500,
                )
            else:
                NotificationOverlay(
                    "Failed to import model!\nPlease try again.", self, timeout=1500
                )

        elif format == "ncnn":
            binFileFilter = "NCNN Bin (*.bin)"
            modelBinFile, _ = QFileDialog.getOpenFileName(
                parent=self,
                caption="Select NCNN Bin",
                dir=self.homeDir,
                filter=binFileFilter,
            )
            if getCustomModelScale(os.path.basename(modelBinFile)):
                if modelBinFile == "":
                    RegularQTPopup("Please select a bin file!")
                    return
                modelParamFile, _ = QFileDialog.getOpenFileName(
                    parent=self,
                    caption="Select NCNN Param",
                    dir=os.path.dirname(modelBinFile),
                    filter=os.path.basename(modelBinFile).replace(".bin", ".param"),
                )
                if modelParamFile == "":
                    RegularQTPopup("Please select a param file!")
                    return
                outputModelFolder = os.path.join(
                    CUSTOM_MODELS_PATH,
                    os.path.basename(modelBinFile).replace(".bin", ""),
                )
                FileHandler.createDirectory(outputModelFolder)
                outputBinPath = os.path.join(
                    outputModelFolder, os.path.basename(modelBinFile)
                )
                FileHandler.copyFile(modelBinFile, outputModelFolder)
                outputParamPath = os.path.join(
                    outputModelFolder, os.path.basename(modelParamFile)
                )
                FileHandler.copyFile(modelParamFile, outputModelFolder)

                if os.path.isfile(outputBinPath) and os.path.isfile(outputParamPath):
                    NotificationOverlay(
                        "Model imported successfully!\nPlease restart the app for the changes to take effect.",
                        self,
                        timeout=1500,
                    )

                else:
                    NotificationOverlay(
                        "Failed to import model!\nPlease try again.", self, timeout=1500
                    )

            else:
                NotificationOverlay(
                    "Custom model does not have a valid\nupscale factor in the name.\nExample: 2x or x2. Skipping import...",
                    self,
                    timeout=1500,
                )

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
        self.outputFileText.setText(
            os.path.join(outputFolder, self.setDefaultOutputFile(outputFolder))
        )

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "",
            "Are you sure you want to exit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,  # type: ignore
        )
        if reply == QMessageBox.Yes:  # type: ignore
            self.processTab.killRenderProcess()
            event.accept()
        else:
            event.ignore()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # setting the pallette

    app.setPalette(Palette())
    window = MainWindow()
    if "--fullscreen" in sys.argv:
        window.showFullScreen()
    window.show()
    sys.exit(app.exec())


"""
custom command args
--debug: runs the app in debug mode
--fullscreen: runs the app in fullscreen
--swap-flatpak-checks: swaps the flatpak checks, ex if the app is running in flatpak, it will run as if it is not
"""

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    if "--debug" in sys.argv:
        import trace

        tracer = trace.Trace(
            ignoredirs=[sys.prefix, sys.exec_prefix], trace=True, count=False
        )
        tracer.run("main()")
    else:
        main()
