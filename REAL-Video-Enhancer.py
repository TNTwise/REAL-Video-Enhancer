from src.constants import CUSTOM_MODELS_PATH, MODELS_PATH, CWD, LOCKFILE, IS_INSTALLED, TEMP_DOWNLOAD_PATH, USE_LOCAL_BACKEND, PLATFORM
import os
try: 
    os.makedirs(CWD) if not os.path.exists(CWD) else None
    # os.chdir(CWD) # need to actually chdir into the directory to have everything run correctly
except:
    pass
import sys
import os
import time
os.environ["PYTHONNOUSERSITE"] = "1" # Prevents python from installing packages in user site
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP"] = "0"
os.environ["PYTHON_JIT"] = "1" # enable python's experimental JIT for better performance in python 3.13
from PySide6.QtCore import QLockFile
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
from src.ModelHandler import getModels, getModelDisplayName

# other imports
from src.Util import (
    getOSInfo,
    getRAMAmount,
    getCPUInfo,
    getAvailableDiskSpace,
    FileHandler,
    log,
    createDirectory
)

createDirectory(os.path.join(CWD, "python"))
createDirectory(os.path.join(CWD, "bin"))


from src.DownloadModels import DownloadModel
from src.DownloadDeps import Dependency, Python, DownloadDependencies
from src.ui.ProcessTab import ProcessTab
from src.ui.DownloadTab import DownloadTab
from src.ui.SettingsTab import SettingsTab, Settings
from src.ui.HomeTab import HomeTab
from src.Backendhandler import BackendHandler
from src.ModelHandler import totalModels
from src.ui.AnimationHandler import AnimationHandler
from src.ui.QTstyle import Palette
from src.ui.QTcustom import RegularQTPopup, NotificationOverlay, IndependentQTPopup
from src.ui.RenderQueue import RenderQueue, RenderOptions
from src.VideoInfo import VideoLoader

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
        self.anyBackendsInstalled = True
        self.videoLength = 1
        self.batchVideos = []


        settings = Settings()
        self.settings = settings


        log(str(os.environ))
        # setup application
        FileHandler.createDirectory(TEMP_DOWNLOAD_PATH)

        # Set up the user interface from Designer.
        start_time = time.time()
        self.setupUi(self)
        end_time = time.time()
        log("Setup ui time: " + str(end_time - start_time))
        #self.VideoPreview.setVisible(False)

        # remove false hope
        self.directMLBackendInstallerContainer.setVisible(False)

        start_time = time.time()
        backendHandler = BackendHandler(self, self.settings)
        end_time = time.time()
        log("Backend handler time: " + str(end_time - start_time))

        start_time = time.time()
        self.renderQueue = RenderQueue(self.renderQueueListWidget)
        end_time = time.time()
        log("Render queue time: " + str(end_time - start_time))

        start_time = time.time()
        if not IS_INSTALLED:
            for dep in Dependency.__subclasses__():
                d = dep()
                d.download()
        end_time = time.time()
        log("Dependency download time: " + str(end_time - start_time))
        
        #popupthread = create_independent_process(IndependentQTPopup, "Checking for dependency updates...")
        #popupthread.start() 

        start_time = time.time()
        for dep in Dependency.__subclasses__():
            d = dep()
            if d.get_if_update_available():
        #        popupthread.terminate()
                d.update_if_updates_available()
        end_time = time.time()
        log("Dependency update time: " + str(end_time - start_time))
        #try:
        #    popupthread.terminate()
        #except Exception:
        #    pass

        start_time = time.time()
        self.backends, self.fullOutput = (
            backendHandler.getAvailableBackends()
        )
        end_time = time.time()

        

        # set default home page
        self.stackedWidget.setCurrentIndex(0)

        self.anyBackendsInstalled = len(self.backends) > 0
        if not self.anyBackendsInstalled:
            self.processBtn.setEnabled(False) # disable process button if no backends are available
            self.stackedWidget.setCurrentIndex(4)
            self.homeBtn.setChecked(False)
            self.downloadBtn.setChecked(True)
            self.processBtn.setToolTip("Please install at least one backend to enable processing.")
            self.processBtn.setToolTipDuration(0)



        
        icon_path = ":/icons/icons/logo-v2.svg"
        self.setWindowIcon(QIcon(icon_path))
        QApplication.setWindowIcon(QIcon(icon_path))
        self.setWindowTitle("REAL Video Enhancer")
        self.setPalette(QApplication.style().standardPalette())
        self.setMinimumSize(950, 600)

        self.aspect_ratio = self.width() / self.height()

        self.QConnect()
        # set up tabs
        self.backendComboBox.addItems(self.backends)
        printOut = (
            "System Information:\n"
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
            + "\nSoftware Information:\n"
            + f"REAL Video Enhancer Version: {version}\n"
            + f"Python Version: {Python().get_version()}\n"
            + f"Utilizing local backend: {USE_LOCAL_BACKEND}\n"
            + f"Working Directory: {CWD}\n"
            + self.fullOutput
        )
        self.systemInfoText.setText(printOut)
        log(printOut)

        # process the output
        total_ncnn_gpus = -1
        total_pytorch_gpus = -1
        halfPrecisionSupport = False
        for line in self.fullOutput.lower().split("\n"):
            if "half precision support:" in line:
                halfPrecisionSupport = "true" in line
            if "ncnn gpu " in line:  # this is to grab every line with "GPU "
                total_ncnn_gpus += 1
            if "pytorch gpu " in line:
                total_pytorch_gpus += 1

        total_pytorch_gpus = max(0, total_pytorch_gpus)  # minimum gpu id is 0
        total_ncnn_gpus = max(0, total_ncnn_gpus)

        if self.anyBackendsInstalled:
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
        downloadDeps = DownloadDependencies(False)
        self.downloadTab.hideUninstallButtons()
        self.downloadTab.showUninstallButton(self.backends)

        # Startup Animation
        self.animationHandler = AnimationHandler()
        self.animationHandler.fadeInAnimation(self)

        if not self.anyBackendsInstalled:
            RegularQTPopup("Welcome to REAL Video Enhancer!\nPlease install at least one backend to get started.")

        # these additions are not ready for stable rve just yet.
        #self.denoiseContainer.setVisible(False)
        self.deblurContainer.setVisible(False)
        #self.denoiseCheckBoxContainer.setVisible(False)
        self.deblurCheckBoxContainer.setVisible(False)

        #player = QMediaPlayer()
        #player.setSource(QUrl.fromLocalFile(r"C:\Users\tntwi\Downloads\CodeGeass-OP3.webm"))
        #player.setVideoOutput(self.VideoPreview)
        #self.VideoPreview.show()
        #self.playbutton.clicked.connect(lambda: player.play())


    def QConnect(self):
        # connect buttons to switch menus
        self.homeBtn.clicked.connect(self.switchToHomePage)
        self.processBtn.clicked.connect(self.switchToProcessingPage)
        self.settingsBtn.clicked.connect(self.switchToSettingsPage)
        self.downloadBtn.clicked.connect(self.switchToDownloadPage)
        self.renderPreviewBtn.clicked.connect(self.renderPreview)
        self.VideoPreview.setVisible(False)
        self.RenderedPreviewControlsContainer.setVisible(False)
        # connect getting default output file

    def renderPreview(self):
        self.renderQueue.clear()
        renderOptions = self.getCurrentRenderOptions()
        renderOptions.outputPath = os.path.join(TEMP_DOWNLOAD_PATH, f"{os.path.basename(renderOptions.inputFile)}_preview.mkv")
        renderOptions.startTime = self.startTimeSpinBox.value()
        FileHandler().removeFile(renderOptions.outputPath)
        renderOptions.endTime = self.endTimeSpinBox.value()
        if renderOptions.endTime <= renderOptions.startTime:
            NotificationOverlay("End time must be greater than start time!", self, timeout=1500)
            return


        renderOptions.isPreview = True
        if renderOptions:
            self.renderQueue.add(renderOptions)
            self.startRender(self.renderQueue)


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
        self.settings.readSettings()
        if self.isVideoLoaded:
            upscaleModelName = self.upscaleModelComboBox.currentText()
            interpolateModelName = self.interpolateModelComboBox.currentText()
            interpolateTimes = self.getInterpolationMultiplier(interpolateModelName)
            scale = self.getUpscaleModelScale(upscaleModelName)
            new_bitrate = 8 if "10" not in self.settingsTab.in_pix_fmt else 10
            inputText = (
                f"FPS: {round(self.videoFps, 0)} -> {round(self.videoFps * interpolateTimes, 0)}\n"
                + f"Resolution: {self.videoWidth}x{self.videoHeight} -> {self.videoWidth * scale}x{self.videoHeight * scale}\n"
                + f"Frame Count: {self.videoFrameCount} -> {int(round(self.videoFrameCount * interpolateTimes, 0))}\n"
                + f"Encoder: {self.videoEncoder} -> {self.settings.settings['encoder']}\n"
                + f"Container: {self.videoContainer} -> {self.settings.settings['video_container']}\n"
                + f"Color Space: {self.colorSpace} -> {self.colorSpace}\n"
                + f"Pixel Format: {self.settingsTab.in_pix_fmt} -> {self.settingsTab.out_pixel_fmt}\n"
                + f"HDR: {self.videoHDR} -> {self.videoHDR if self.settings.settings['auto_hdr_mode'] == 'True' else 'False'}\n"
                + f"Bit Depth: {self.videoBitDepth} bit -> {new_bitrate if self.settings.settings['auto_hdr_mode'] == 'True' else 8} bit\n"
            )
            
            self.inputVideoInfoTextEdit.setFontPointSize(10)
            self.inputVideoInfoTextEdit.setText(inputText)

    def getInterpolationMultiplier(self, interpolateModelName):
        if interpolateModelName == "None" or not self.interpolateCheckBox.isChecked():
            interpolateTimes = 1
        else:
            interpolateTimes = self.interpolationMultiplierSpinBox.value()
        return interpolateTimes

    def getUpscaleModelScale(self, upscaleModelName):
        if upscaleModelName == "None" or upscaleModelName == "" or not self.upscaleCheckBox.isChecked():
            scale = 1
        else:
            scale = int(self.upscaleScaleSpinBox.value())
        return scale

    def setDefaultOutputFile(self, inputFile, outputDirectory):
        """
        Sets the default output file for the video enhancer.
        Parameters:
        - useDefaultVideoPath (bool): Flag indicating whether to use the default video path for the output file.
        Returns:
        None
        """

        # check if there is a video loaded
        if self.isVideoLoaded:
            if inputFile.strip().replace(" ", "") == "{MULTIPLE_FILES}":
                inputFile = " { FILE_NAME } "

            upscaleModelName = self.upscaleModelComboBox.currentText()
            decompressModelName = self.decompressModelComboBox.currentText()
            denoiseModelName = self.denoiseModelComboBox.currentText()
            interpolateModelName = self.interpolateModelComboBox.currentText()
            
            interpolateTimes = self.getInterpolationMultiplier(interpolateModelName)
            scale = self.getUpscaleModelScale(upscaleModelName)
            container = self.settings.settings["video_container"]

            file_name = os.path.splitext(os.path.basename(inputFile))[0]
            base_file_name = (
                f"{file_name}"
                + ("" if not self.interpolateCheckBox.isChecked() else f"_{getModelDisplayName(interpolateModelName)}")
                + ("" if not self.decompressCheckBox.isChecked() else f"_{getModelDisplayName(decompressModelName)}")
                + ("" if  not self.denoiseCheckBox.isChecked() else f"_{getModelDisplayName(denoiseModelName)}")
                + ("" if  not self.upscaleCheckBox.isChecked() else f"_{getModelDisplayName(upscaleModelName)}")
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
        isInterpolate = self.interpolateCheckBox.isChecked()
        isUpscale = self.upscaleCheckBox.isChecked()
        isDeblur = self.deblurCheckBox.isChecked()
        isDenoise = self.denoiseCheckBox.isChecked()
        isDecompress = self.decompressCheckBox.isChecked()
        
        self.interpolationContainer.setVisible(isInterpolate)
        self.interpolateContainer_2.setVisible(isInterpolate)
        self.deblurContainer.setVisible(isDeblur)
        self.denoiseContainer.setVisible(isDenoise)
        self.decompressContainer.setVisible(isDecompress)
        
        # disable decompress for now
        #self.decompressCheckBoxContainer.setVisible(False)
        #self.decompressContainer.setVisible(False)
        
        # set interpolation container visible if interpolate model is not none
        self.upscaleContainer.setVisible(isUpscale or isDeblur or isDenoise or isDecompress)
        self.generalUpscaleContainer.setVisible(isUpscale)
        self.settings.readSettings()
        self.setDefaultOutputFile(self.inputFileText.text(), self.settings.settings["output_folder_location"])
        self.updateVideoGUIText()
        self.startTimeSpinBox.setMaximum(self.videoLength)
        self.endTimeSpinBox.setMaximum(self.videoLength)
        self.timeInVideoScrollBar.setMaximum(self.videoLength)
        if isUpscale and (self.upscaleModelComboBox.currentText() != "" or self.upscaleModelComboBox.currentText() != "None"):
            try:
                max_scale = totalModels[self.upscaleModelComboBox.currentText()][2]
                self.upscaleScaleSpinBox.setMaximum(max_scale if max_scale > 0 else 4)
            except KeyError: # idk why it does this, gui is shit tbh.
                self.upscaleScaleSpinBox.setMaximum(4)

    def getCurrentRenderOptions(self, input_file=None, output_path=None):
        interpolate = self.interpolateModelComboBox.currentText()
        upscale = self.upscaleModelComboBox.currentText()
        deblur = self.deblurModelComboBox.currentText()
        denoise = self.denoiseModelComboBox.currentText()
        decompress = self.decompressModelComboBox.currentText()
        input_file = self.inputFileText.text() if input_file is None else input_file
        output_path = self.outputFileText.text() if output_path is None else output_path
        interpolateModelFile = None
        upscaleModelFile = None
        deblurModelFile = None
        denoiseModelFile = None
        decompressModelFile = None
        if not self.interpolateCheckBox.isChecked():
            interpolate = None
        if not self.upscaleCheckBox.isChecked():
            upscale = None
        if not self.deblurCheckBox.isChecked():
            deblur = None
        if not self.denoiseCheckBox.isChecked():
            denoise = None
        if not self.decompressCheckBox.isChecked():
            decompress = None
        if not self.isVideoLoaded:
            NotificationOverlay("Video is not loaded!", self, timeout=1500)
            return 1

        if not interpolate and not upscale and not deblur and not denoise and not decompress:
            NotificationOverlay("Please select at least one model!", self, timeout=1500)
            return 1

        backend = self.backendComboBox.currentText()
        upscaleModelArch = "custom"
        interpolateModels, upscaleModels, deblurModels, denoiseModels, decompressModels = getModels(backend)

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
                return 1
        upscaleTimes = 1
        modelScale = 1
        if upscale:
            upscaleModelFile = upscaleModels[upscale][0]
            upscaleDownloadFile = upscaleModels[upscale][1]
            modelScale = upscaleModels[upscale][2]
            upscaleTimes = int(self.upscaleScaleSpinBox.value())
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
                    return 1
        if deblur:
            deblurModelFile = deblurModels[deblur][0]
            deblurDownloadFile = deblurModels[deblur][1]
            dm = DownloadModel(
                modelFile=deblurModelFile,
                downloadModelFile=deblurDownloadFile,
            )
            if not dm.downloadModel():
                NotificationOverlay(
                    "Unable to add to render queue.\nModel can't be downloaded.\nPlease check your network and try again.",
                    self,
                    timeout=2500,
                )
                return 1
        
        if denoise:
            denoiseModelFile = denoiseModels[denoise][0]
            denoiseDownloadFile = denoiseModels[denoise][1]
            dm = DownloadModel(
                modelFile=denoiseModelFile,
                downloadModelFile=denoiseDownloadFile,
            )
            if not dm.downloadModel():
                NotificationOverlay(
                    "Unable to add to render queue.\nModel can't be downloaded.\nPlease check your network and try again.",
                    self,
                    timeout=2500,
                )
                return 1
        if decompress:
            decompressModelFile = decompressModels[decompress][0]
            decompressDownloadFile = decompressModels[decompress][1]
            dm = DownloadModel(
                modelFile=decompressModelFile,
                downloadModelFile=decompressDownloadFile,
            )
            if not dm.downloadModel():
                NotificationOverlay(
                    "Unable to add to render queue.\nModel can't be downloaded.\nPlease check your network and try again.",
                    self,
                    timeout=2500,
                )
                return 1
        hdrmode = False
        """if "gmfss" in interpolateDownloadFile.lower():
            log("GMFSS model detected, enabling rgb48 proc to have correct colors.")
            hdrmode = True"""
        if self.settings.settings["auto_hdr_mode"] == "True" and (self.videoHDR or self.videoBitDepth > 8):
            hdrmode = True
        return RenderOptions(
            inputFile=input_file,
            outputPath=output_path,
            videoWidth=self.videoWidth,
            videoHeight=self.videoHeight,
            videoFps=self.videoFps,
            tilingEnabled=self.tilingCheckBox.isChecked(),
            tilesize=self.tileSizeComboBox.currentText(),
            videoFrameCount=self.videoFrameCount,
            backend=self.backendComboBox.currentText(),
            interpolateTimes=self.getInterpolationMultiplier(
                self.interpolateModelComboBox.currentText()
            ),
            benchmarkMode=self.benchmarkModeCheckBox.isChecked(),
            sloMoMode=self.sloMoModeCheckBox.isChecked(),
            dyanmicScaleOpticalFlow=self.dynamicScaledOpticalFlowCheckBox.isChecked(),
            ensemble=self.ensembleCheckBox.isChecked(),
            modelScale=modelScale,
            upscaleModelArch=upscaleModelArch,
            upscaleModelFile=upscaleModelFile,
            deblurModelFile=deblurModelFile,
            denoiseModelFile=denoiseModelFile,
            decompressModelFile=decompressModelFile,
            interpolateModelFile=interpolateModelFile,
            hdrMode=hdrmode,
            overrideUpscaleScale=upscaleTimes,
            encoderCommand=self.EncoderCommand.text(),
        )

    def addToRenderQueue(self):
        self.settings.readSettings()
        input_file = self.inputFileText.text()
        output_path = self.outputFileText.text()

        if "{MULTIPLE_FILES}" in input_file.strip().replace(" ", ""):
            output_path = os.path.dirname(output_path)
            for video in self.batchVideos:
                videoHandler = VideoLoader(video)
                videoHandler.loadVideo()
                videoHandler.getData()
                self.videoWidth = videoHandler.width
                self.videoHeight = videoHandler.height
                self.videoFps = videoHandler.fps
                self.videoLength = videoHandler.duration
                self.videoFrameCount = videoHandler.total_frames
                self.videoEncoder = videoHandler.codec_str
                self.videoBitrate = videoHandler.bitrate
                self.videoContainer = videoHandler.videoContainer
                self.colorSpace = videoHandler.color_space
                self.pixelFMT = videoHandler.pixel_format
                self.videoHDR = videoHandler.is_hdr
                self.videoBitDepth = videoHandler.bit_depth
                

                # set output_path for checking
                default_output_path = self.setDefaultOutputFile(
                    video, output_path
                )

                # check if file already exists in renderQueue
                if any(item.outputPath == default_output_path for item in self.renderQueue.getQueue()):
                    NotificationOverlay(f"Skipped (Already in queue): {default_output_path}", self, timeout=1500)
                    continue

                renderOptions = self.getCurrentRenderOptions(
                    input_file=video,
                    output_path=self.setDefaultOutputFile(video, output_path),
                )
                if renderOptions == 1:
                    return
                # alert user that item has been added to queue
                self.renderQueue.add(renderOptions)
                

            # clear batch file list
            self.batchVideos = []
            return


        for renderOptions in self.renderQueue.getQueue():
            if output_path == renderOptions.outputPath:
                NotificationOverlay("Output file already in queue!", self, timeout=1500)
                return

        renderOptions = self.getCurrentRenderOptions()
        if renderOptions == 1:
            return
        # alert user that item has been added to queue
        NotificationOverlay(
            f"{self.inputFileText.text()} Added to queue!", self, timeout=1500
        )
        self.renderQueue.add(renderOptions)

    def startRender(self, renderQueue=None):
        if not renderQueue:
            renderQueue = self.renderQueue
        if len(renderQueue.queue) == 0:
            NotificationOverlay("Render queue is empty!", self, timeout=1500)
            return
        self.startRenderButton.setEnabled(False)
        self.VideoPreview.setVisible(False)
        self.previewLabel.setVisible(True)

        self.disableProcessPage()
        self.processTab.run(renderQueue)

    def disableProcessPage(self):
        for child in self.generalSettings.children():
            child.setEnabled(False)
        for child in self.advancedSettings.children():
            child.setEnabled(False)
        for child in self.renderQueueTab.children():
            child.setEnabled(False)
        for child in self.encoderSettings.children():
            child.setEnabled(False)
        self.RenderedPreviewControlsContainer.setEnabled(False)
        self.scrollArea_4.setEnabled(True)
        self.scrollAreaWidgetContents_4.setEnabled(False)
        self.widget_5.setEnabled(True)

    def enableProcessPage(self):
        for child in self.generalSettings.children():
            child.setEnabled(True)
        for child in self.advancedSettings.children():
            child.setEnabled(True)
        for child in self.renderQueueTab.children():
            child.setEnabled(True)
        for child in self.encoderSettings.children():
            child.setEnabled(True)
        self.RenderedPreviewControlsContainer.setEnabled(True)
        self.scrollAreaWidgetContents_4.setEnabled(True)

    def loadVideo(self, inputFile, multi_file=False):
        if "{MULTIPLE_FILES}" in inputFile.strip().replace(" ", ""):
            return
        if inputFile == "":
            NotificationOverlay("Please select a video file!", self, timeout=1500)
            return
        if multi_file:
            self.outputFileText.setEnabled(False)
            for file in os.listdir(inputFile):
                video = os.path.join(inputFile, file)
                videoHandler = VideoLoader(video)
                videoHandler.loadVideo()
                if videoHandler.isValidVideo():
                   self.batchVideos.append(video)
            if self.batchVideos == 0:
                NotificationOverlay("No valid videos found in the selected folder!", self, timeout=1500)
                return
            NotificationOverlay("Loaded " + str(len(self.batchVideos)) + " videos.", self, timeout=1500)
            self.inputFileText.setText(" { MULTIPLE_FILES } ")
            self.inputFileText.setEnabled(False)
            self.videoWidth = 0
            self.videoHeight = 0
            self.videoFps = 0
            self.videoLength = 0
            self.videoFrameCount = 0
            self.videoEncoder = "Multi File"
            self.videoBitrate = "Multi File"
            self.videoContainer = "Multi File"
            self.colorSpace = "Multi File"
            self.pixelFMT = "Multi File"
            self.videoHDR = "Multi File"
            self.videoBitDepth = "Multi File"
        else:
                    
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
            self.colorSpace = videoHandler.color_space
            self.pixelFMT = videoHandler.pixel_format
            self.videoHDR = videoHandler.is_hdr
            self.videoBitDepth = videoHandler.bit_depth

            self.inputFileText.setText(inputFile)
            self.outputFileText.setEnabled(True)

        self.outputFileSelectButton.setEnabled(True)
        self.openOutputFolderButton.setEnabled(True)    
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

        fileFilter = "Video files (*.mp4 *.mov *.webm *.mkv);;All Files (*)"
        inputFile, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select File",
            dir=self.settings.settings["last_input_folder_location"] if os.path.exists(self.settings.settings["last_input_folder_location"]) else self.homeDir,
            filter=fileFilter,
        )
        self.loadVideo(inputFile)
        self.settings.writeSetting("last_input_folder_location", str(os.path.dirname(inputFile)))
    
    def openBatchFiles(self):
        inputFolder = QFileDialog.getExistingDirectory(
            self,
            caption="Select Input Directory",
            dir=self.homeDir,
        )
        self.loadVideo(inputFolder, multi_file=True)

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
               
            if modelFile == "":
                RegularQTPopup("Please select a model file!")
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

       
    # output file button
    def openOutputFolder(self):
        """
        Opens home folder or the same folder as the input file only if the input file is a single file,
        sets the directory that is selected to the self.outputFolder variable
        sets the outputFileText to the output directory

        It will also read the input file name, and generate an output file based on it.
        """
        outputFolder = QFileDialog.getExistingDirectory(
            self,
            caption="Select Output Directory",
            dir=str(os.path.dirname(self.inputFileText.text())) if (self.isVideoLoaded and len(self.batchVideos) == 0 and os.path.exists(os.path.dirname(self.inputFileText.text()))) else self.homeDir,
        )
        self.outputFileText.setText(
            os.path.join(outputFolder, self.setDefaultOutputFile(self.inputFileText.text(), outputFolder))
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
            FileHandler.removeFolder(TEMP_DOWNLOAD_PATH)
            if hasattr(self, "processTab"):
                self.processTab.killRenderProcess()
            event.accept()
        else:
            event.ignore()



def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(Palette())

    """if not "--unlock" in sys.argv:
        lock_file = QLockFile(LOCKFILE)
        if not lock_file.tryLock(10):
            QMessageBox.warning(None, "Instance Running", "Another instance is already running.")
            sys.exit(0)"""

    # setting the pallette
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
