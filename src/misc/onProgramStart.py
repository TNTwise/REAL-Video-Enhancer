from src.programData.settings import *
from src.misc.messages import *
import src.programData.checks as checks
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices, QIntValidator, QIcon
import os
from src.misc.log import *
from src.getLinkVideo.get_video import *
from src.getModels.rifeModelsFunctions import rife_cuda_checkboxes
from src.programData.version import returnVersion
import site

def open_link(urll):
    url = QUrl(urll)
    QDesktopServices.openUrl(url)


def bindButtons(self):
    settings = Settings()
    self.ui.RifeSettings.clicked.connect(
        lambda: src.getModels.get_models_settings.get_rife(self)
    )
    self.ui.esrganHelpModel.clicked.connect(
        lambda: open_link(
            "https://github.com/upscayl/upscayl/wiki/Model-Conversion-Guide"
        )
    )
    self.ui.cudaArchSupportButton.clicked.connect(
        lambda: open_link(
            r"https://github.com/chaiNNer-org/spandrel?tab=readme-ov-file#model-architecture-support"
        )
    )

    self.ui.discordButton.clicked.connect(
        lambda: open_link(r"https://discord.gg/mRReVBMQtN")
    )
    self.ui.githubButton.clicked.connect(
        lambda: open_link(r"https://github.com/TNTwise/REAL-Video-Enhancer")
    )
    self.ui.customModelsButton.clicked.connect(
        lambda: open_link(r"https://openmodeldb.info/")
    )

    self.ui.Rife_Times.currentIndexChanged.connect(self.showChangeInFPS)
    self.ui.gpuThreadingSpinBox.valueChanged.connect(self.changeVRAM)
    self.ui.gpuIDSpinBox.valueChanged.connect(lambda: change_GPU_ID(self))
    self.ui.UHDResCutoffSpinBox.valueChanged.connect(lambda: change_UHD_res(self))
    self.ui.frameIncrementSpinBox.valueChanged.connect(
        lambda: selFrameIncrements(self.ui.frameIncrementSpinBox.value())
    )
    self.ui.AICombo.currentIndexChanged.connect(self.switchUI)
    self.ui.modeCombo.currentIndexChanged.connect(self.switchMode)
    self.ui.DiscordRPCBox.stateChanged.connect(lambda: changeDiscordRPC(self))
    self.ui.sceneChangeLineEdit.textChanged.connect(lambda: changeSceneDetection(self))
    self.ui.frameIncrementsModeCombo.currentTextChanged.connect(
        lambda: selFrameIncrementsMode(self)
    )
    self.ui.sceneChangeDetectionCheckBox.stateChanged.connect(
        lambda: selSceneDetectionMode(self)
    )
    self.ui.halfPrecisionCheckBox.stateChanged.connect(lambda: halfPrecision(self))
    self.ui.RenderDirButton.clicked.connect(lambda: selRenderDir(self))

    self.ui.Input_video_rife.clicked.connect(
        lambda: self.openFileNameDialog(
            "Video", [".mp4", ".mkv", ".webm", ".mov", ".webm", "avi"]
        )
    )

    self.ui.Output_folder_rife.clicked.connect(self.openFolderDialog)

    # link buttons
    self.ui.UHDModeHelpButton.clicked.connect(lambda: UHDModeHelp(self))
    self.ui.gpuIDHelpButton.clicked.connect(lambda: GPUIDHelp(self))
    self.ui.frameIncrementHelp.clicked.connect(lambda: frame_increments_help(self))
    self.ui.SettingsMenus.clicked.connect(self.settings_menu)
    self.ui.OutputDirectoryButton.clicked.connect(lambda: selOutputDir(self))
    self.ui.resetSettingsButton.clicked.connect(self.restore_default_settings)
    self.ui.gpuThreadingHelpButton.clicked.connect(lambda: vram_help(self))
    self.ui.logButton.clicked.connect(lambda: viewLogs(self))
    self.ui.RifeResume.clicked.connect(self.resume_render)
    self.ui.sceneChangeSensativityButton.clicked.connect(
        lambda: show_scene_change_help(self)
    )
    self.ui.encoderHelpButton.clicked.connect(lambda: encoder_help(self))
    self.ui.ensembleHelpButton.clicked.connect(lambda: ensemble_help(self))
    self.ui.renderTypeCombo.currentIndexChanged.connect(lambda: selRenderType(self))
    self.ui.renderTypeHelpButton.clicked.connect(lambda: render_help(self))
    self.ui.Rife_Model.currentIndexChanged.connect(self.greyOutRifeTimes)
    self.ui.imageHelpButton.clicked.connect(lambda: image_help(self))
    self.ui.halfPrecisionHelpButton.clicked.connect(lambda: halfPrecision_help(self))
    self.ui.GMFSSHelpButton.clicked.connect(lambda: gmfss_help(self))
    self.ui.imageComboBox.currentIndexChanged.connect(
        lambda: settings.change_setting(
            "Image_Type", f"{self.ui.imageComboBox.currentText()}"
        )
    )
    self.ui.sceneChangeMethodComboBox.currentIndexChanged.connect(
        lambda: selSceneDetectionMethod(self)
    )
    self.ui.EncoderCombo.currentIndexChanged.connect(lambda: selEncoder(self))
    # apparently adding multiple currentindexchanged causes a memory leak unless i sleep, idk why it does this but im kinda dumb
    self.ui.ESRGANModelSelectButton.clicked.connect(
        lambda: self.openFileNameDialog("NCNN Model", [".bin"])
    )
    self.ui.PyTorchModelSelectButton.clicked.connect(
        lambda: self.openFileNameDialog("CUDA Model", [".pkl", ".pth", ".pt"])
    )

    self.ui.Input_video_rife_url.clicked.connect(lambda: get_linked_video(self))
    self.ui.VidQualityCombo.currentIndexChanged.connect(lambda: selVidQuality(self))
    self.ui.defaultRifeModel.currentIndexChanged.connect(
        lambda: settings.change_setting(
            "DefaultRifeModel", f"{self.ui.defaultRifeModel.currentText()}"
        )
    )
    self.ui.InstallButton.clicked.connect(
        lambda: src.getModels.get_models_settings.run_install_models_from_settings(self)
    )
    self.ui.Rife_Times.currentIndexChanged.connect(
        lambda: self.showChangeInFPS(self.localFile)
    )
    try:
        from notify import notification

        self.ui.NotificationsCheckBox.stateChanged.connect(lambda: Notifications(self))
    except Exception as e:
        self.ui.NotificationsCheckBox.hide()
        settings.change_setting("Notifications", "Disabled")
    self.ui.uninstallButton.clicked.connect(lambda: uninstallAPP(self))

    helpButtons = [
        self.ui.sceneChangeSensativityButton,
        self.ui.encoderHelpButton,
        self.ui.imageHelpButton,
        self.ui.gpuThreadingHelpButton,
        self.ui.renderTypeHelpButton,
        self.ui.frameIncrementHelp,
        self.ui.esrganHelpModel,
        self.ui.ensembleHelpButton,
        self.ui.UHDModeHelpButton,
        self.ui.gpuIDHelpButton,
        self.ui.halfPrecisionHelpButton,
        self.ui.GMFSSHelpButton,
    ]
    icon = QIcon(f"{thisdir}/icons/Rife-ESRGAN-Video-Settings - Help.png")
    for button in helpButtons:
        button.setIcon(icon)

    # Bind animations
    self.setWindowIcon(QIcon(f"{thisdir}/icons/logo v1.png"))

def settingsStart(self):
    settings = Settings()
    self.settings = Settings()

    self.videoQuality = settings.videoQuality
    self.encoder = settings.Encoder

    if settings.SceneChangeMethod == "ffmpeg":
        self.ui.sceneChangeMethodComboBox.setCurrentIndex(0)

    if settings.SceneChangeMethod == "pyscenedetect":
        self.ui.sceneChangeMethodComboBox.setCurrentIndex(1)
    if settings.HalfPrecision == "False":
        self.ui.halfPrecisionCheckBox.setChecked(False)
    else:
        self.ui.halfPrecisionCheckBox.setChecked(True)
    self.ui.sceneChangeLineEdit.setText(settings.SceneChangeDetection[2])
    if self.encoder == "264":
        self.ui.EncoderCombo.setCurrentIndex(0)
    if self.encoder == "265":
        self.ui.EncoderCombo.setCurrentIndex(1)
    if self.encoder == "VP9":
        self.ui.EncoderCombo.setCurrentIndex(2)
    if self.encoder == "AV1":
        self.ui.EncoderCombo.setCurrentIndex(3)
    if self.encoder == "ProRes":
        self.ui.EncoderCombo.setCurrentIndex(4)
    if self.encoder == "Lossless":
        self.ui.EncoderCombo.setCurrentIndex(5)
    if self.videoQuality == "10":
        self.ui.VidQualityCombo.setCurrentText("Lossless")
    if self.videoQuality == "14":
        self.ui.VidQualityCombo.setCurrentText("Very High")
    if self.videoQuality == "18":
        self.ui.VidQualityCombo.setCurrentText("High")
    if self.videoQuality == "20":
        self.ui.VidQualityCombo.setCurrentText("Medium")
    if self.videoQuality == "22":
        self.ui.VidQualityCombo.setCurrentText("Low")
    if settings.RenderType == "Classic":
        self.ui.renderTypeCombo.setCurrentIndex(0)
    elif settings.RenderType == "Optimized":
        self.ui.renderTypeCombo.setCurrentIndex(1)
    else:
        self.ui.renderTypeCombo.setCurrentIndex(2)

    if settings.SceneChangeDetectionMode == "Enabled":
        self.ui.sceneChangeDetectionCheckBox.setChecked(True)
        self.ui.label_3.show()
        self.ui.sceneChangeSensativityButton.show()
        self.ui.sceneChangeLineEdit.show()
    else:
        self.ui.sceneChangeDetectionCheckBox.setChecked(False)
        self.ui.label_3.hide()
        self.ui.sceneChangeSensativityButton.hide()
        self.ui.sceneChangeLineEdit.hide()

    if os.path.exists(f"{settings.RenderDir}") == False:
        settings.change_setting("RenderDir", f"{thisdir}")

    if settings.Notifications == "Enabled":
        self.ui.NotificationsCheckBox.setChecked(True)
    else:
        self.ui.NotificationsCheckBox.setChecked(False)

    if settings.DiscordRPC == "Enabled":
        self.ui.DiscordRPCBox.setChecked(True)

    else:
        self.ui.DiscordRPCBox.setChecked(False)

    self.ui.imageComboBox.setCurrentText(f"{settings.Image_Type}")
    self.ui.defaultRifeModel.setCurrentText(f"{settings.DefaultRifeModel}")
    self.ui.OutputDirectoryLabel.setText(settings.OutputDir)
    self.ui.frameIncrementSpinBox.setValue(int(settings.FrameIncrements))
    self.ui.UHDResCutoffSpinBox.setValue(int(settings.UHDResCutOff))
    self.ui.RenderPathLabel.setText(f"{settings.RenderDir}")
    self.ui.gpuIDSpinBox.setValue(int(settings.gpuID))
    self.gpuMemory = settings.VRAM

    self.ui.gpuThreadingSpinBox.setValue(int(self.gpuMemory))


def hideChainModeButtons(self):
    self.ui.TimesInterpolationLabel.hide()
    self.ui.TimesInterpolationComboBox.hide()
    self.ui.TimesUpscaleComboBox.hide()
    self.ui.TimesUpscalingLabel.hide()
    self.ui.ModelInterpolationComboBox.hide()
    self.ui.ModelInterpolationLabel.hide()
    self.ui.ModelUpscalingComboBox.hide()
    self.ui.ModelUpscalingLabel.hide()
    self.ui.AIInterpolationComboBox.hide()
    self.ui.AIInterpolationLabel.hide()
    self.ui.AIUpscalingComboBox.hide()
    self.ui.AIUpscalingLabel.hide()


def cuda_shit(self):
    if checks.isCUDA():  # shit to do on cuda ver
        self.ui.modelTabWidget.setTabEnabled(1, True)
        os.system(f'mkdir -p "{thisdir}/models/custom-models-cuda"')
        # shit to do if cupy
        if checks.isCUPY():
            self.ui.GMFSSCUDACheckBox.setDisabled(False)
        else:
            self.ui.GMFSSCUDACheckBox.setDisabled(True)

    else:
        self.ui.modelTabWidget.setTabEnabled(1, False)

def setupSettings(self):
    self.ui.VideoOptionsFrame.hide()
    self.ui.RenderOptionsFrame.hide()
    self.ui.GeneralOptionsFrame.hide()
    self.ui.SettingsMenus.setCurrentRow(0)
    self.ui.GeneralOptionsFrame.show()

def hideUnusedFeatures(self):
    self.ui.RifeResume.hide()
    self.ui.RifePause.hide()
    self.ui.QueueButton.hide()
    self.ui.QueueListWidget.hide()

def createDirectories():
    os.makedirs(
        os.path.join(f"{thisdir}", "models", "custom_models_ncnn", "models"),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(f"{thisdir}", "models", "custom-models-cuda"), exist_ok=True
    )
def exportTRTlibsToPATH(self):
    print(f'export LD_LIBRARY_PATH={os.getcwd()}/_internal/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH')
    os.environ['LD_LIBRARY_PATH']=f'{os.getcwd()}/_internal/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH'
def onApplicationStart(self):
    # this is kind of a mess
    thisdir = src.programData.thisdir.thisdir()
    log(
        "Program Version: "
        + returnVersion()
        + "\n===================================================================="
    )

    createDirectories()

    self.ui.AICombo.clear()  # needs to be in this order, before SwitchUI is called

    set_model_params(self)
    hideChainModeButtons(self)
    exportTRTlibsToPATH(self)
    # get esrgan models

    self.ui.ESRGANModelSelectButton.show()
    self.ui.label_20.show()
    self.ui.esrganHelpModel.show()
    self.input_file = ""
    
    

    self.switchUI()

    self.ui.installModelsProgressBar.setMaximum(100)

    self.ui.gpuThreadingSpinBox.setMinimum(1)

    # Define Variables
    self.input_file = ""
    self.output_folder = ""

    self.ui.InstallModelsFrame.hide()
    

    cuda_shit(self)

    setupSettings(self)

    hideUnusedFeatures(self)
    os.system(
        "ln -sf {app/com.discordapp.Discord,$XDG_RUNTIME_DIR}/discord-ipc-0"
    )  # Enables discord RPC on flatpak
    
    onlyInt = QIntValidator()
    onlyInt.setRange(0, 9)
    self.ui.sceneChangeLineEdit.setValidator(onlyInt)

    


    
    self.QueueList = []
    self.setDirectories()

    

    

    # call settings specific changes to GUI
    settingsStart(self)

    # enables cuda rife models
    rife_cuda_checkboxes(self)


def set_model_params(self):
    """
    Sets up individual models in gui, and if they exist, adds them to their respective category.


    """
    models_installed = checks.check_for_individual_models()
    self.model_labels = {}
    self.ui.RifeSettings.setEnabled(False)
    for i in models_installed:
        if "Rife-ncnn" == i:
            self.ui.RifeCheckBox.setChecked(True)
            self.model_labels["Rife (NCNN)"] = "interpolation"
            self.ui.RifeSettings.setEnabled(True)
        if "RealESRGAN-ncnn" == i:
            self.ui.RealESRGANCheckBox.setChecked(True)
            self.model_labels["RealESRGAN (NCNN)"] = "upscaling"
        if "RealCUGAN-ncnn" == i:
            self.ui.RealCUGANCheckBox.setChecked(True)
            self.model_labels["RealCUGAN (NCNN)"] = "upscaling"
        if "RealSR-ncnn" == i:
            self.ui.RealSRCheckBox.setChecked(True)
            self.model_labels["RealSR (NCNN)"] = "upscaling"
        if "Waifu2X-ncnn" == i:
            self.ui.Waifu2xCheckBox.setChecked(True)
            self.model_labels["Waifu2X (NCNN)"] = "upscaling"

        if "Vapoursynth-RIFE" == i:
            self.ui.VapoursynthRIFECheckBox.setChecked(True)
            self.model_labels["Vapoursynth-RIFE"] = "interpolation"
        if "IFRNET" == i:
            self.ui.CainCheckBox.setChecked(True)
            self.model_labels["IFRNET (NCNN)"] = "interpolation"

        if "rife-cuda" == i:
            self.ui.RifeCUDACheckBox.setChecked(True)
            self.model_labels["Rife Cuda (Nvidia only)"] = "interpolation"

        if "rife-cuda-trt" == i:
            os.makedirs(
                os.path.join(f"{thisdir}", "models", "rife-trt-engines"), exist_ok=True
            )
            self.model_labels["Rife TensorRT (Nvidia only)"] = "interpolation"

        if "Custom NCNN Models" == i:
            self.model_labels["Custom NCNN models"] = "upscaling"

        if "realesrgan-cuda" == i:
            self.ui.RealESRGANCUDACheckBox.setChecked(True)
            self.model_labels["RealESRGAN Cuda (Nvidia only)"] = "upscaling"

        if "gfmss-cuda" == i:
            self.ui.GMFSSCUDACheckBox.setChecked(True)
            self.model_labels["GMFSS Cuda (Nvidia only)"] = "interpolation"

        if "custom-cuda-models" == i:
            self.model_labels["Custom CUDA models"] = "upscaling"

        if "SPAN (NCNN)" == i:
            self.ui.SPANNCNNCheckBox.setChecked(True)
            self.model_labels["SPAN (NCNN)"] = "upscaling"

    self.ui.modeCombo.clear()
    upscale_list = []
    for i in range(self.ui.modeCombo.count()):
        item_text = self.ui.modeCombo.itemText(i)
        upscale_list.append(item_text)
    if "Interpolation" not in upscale_list:
        for key, value in self.model_labels.items():
            if value == "interpolation":
                self.ui.modeCombo.addItem("Interpolation")
                break
    if "Upscaling" not in upscale_list:
        for key, value in self.model_labels.items():
            if value == "upscaling":
                self.ui.modeCombo.addItem("Upscaling")
                break
    

    self.switchMode()
