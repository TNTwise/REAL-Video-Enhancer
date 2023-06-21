import os
from src.settings import *
from src.messages import *
import src.start as start
settings = Settings()
from PyQt5.QtGui import QIntValidator

def onApplicationStart(self):
    #Define Variables
    self.input_file = ''
    self.output_folder = ''
    self.output_folder = settings.OutputDir 
    self.videoQuality = settings.videoQuality
    self.encoder = settings.Encoder
    if os.path.exists(f"{settings.RenderDir}") == False:
        settings.change_setting('RenderDir',f'{thisdir}')
    self.render_folder = settings.RenderDir
    self.ui.sceneChangeSensativityButton.setIcon(QIcon(f"{thisdir}/icons/Rife-ESRGAN-Video-Settings - Help.png"))
    self.ui.encoderHelpButton.setIcon(QIcon(f"{thisdir}/icons/Rife-ESRGAN-Video-Settings - Help.png"))
    
    
    self.ui.DiscordRPCBox.stateChanged.connect(lambda: changeDiscordRPC(self))
    if settings.DiscordRPC == 'Enabled':
        self.ui.DiscordRPCBox.setChecked(True)
    else:
        self.ui.DiscordRPCBox.setChecked(False)
    
    onlyInt = QIntValidator()
    onlyInt.setRange(0, 9)
    self.ui.sceneChangeLineEdit.setValidator(onlyInt)
    self.ui.sceneChangeLineEdit.textChanged.connect(lambda: changeSceneDetection(self))
    self.ui.sceneChangeLineEdit.setText(settings.SceneChangeDetection[2])
    if self.encoder == '264':
        self.ui.EncoderCombo.setCurrentIndex(0)
    if self.encoder == '265':
        self.ui.EncoderCombo.setCurrentIndex(1)
    if self.videoQuality == '10':
        self.ui.VidQualityCombo.setCurrentText('Lossless')
    if self.videoQuality == '14':
        self.ui.VidQualityCombo.setCurrentText('High')
    if self.videoQuality == '18':
        self.ui.VidQualityCombo.setCurrentText('Medium')
    if self.videoQuality == '22':
        self.ui.VidQualityCombo.setCurrentText('Low')
    self.ui.Rife_Model.currentIndexChanged.connect(self.greyOutRifeTimes)
    self.ui.RealESRGAN_Model.setCurrentIndex(1)
    self.ui.RealESRGAN_Model.currentIndexChanged.connect((self.greyOutRealSRTimes))
    #link help buttons
    self.ui.sceneChangeSensativityButton.clicked.connect(lambda: show_scene_change_help(self))
    self.ui.encoderHelpButton.clicked.connect(lambda:  encoder_help(self))

    self.ui.RenderPathLabel.setText(f"{settings.RenderDir}")
    self.ui.RenderDirButton.clicked.connect(lambda: selRenderDir(self))
    self.ui.verticalTabWidget.setCurrentWidget(self.ui.verticalTabWidget.findChild(QWidget, 'Rife'))
    self.ui.Input_video_rife.clicked.connect(self.openFileNameDialog)
    self.ui.Input_video_RealESRGAN.clicked.connect(self.openFileNameDialog)
    self.ui.Output_folder_rife.clicked.connect(self.openFolderDialog)
    self.ui.Output_folder_RealESRGAN.clicked.connect(self.openFolderDialog)
    self.ui.VideoOptionsFrame.hide()
    self.ui.RenderOptionsFrame.hide()
    self.ui.GeneralOptionsFrame.hide()
    self.ui.RifeStart.clicked.connect(lambda: start.startRife(self))
    self.ui.RealESRGANStart.clicked.connect(lambda: start.startRealSR(self))
    
    self.ui.EncoderCombo.currentIndexChanged.connect(lambda: selEncoder(self))
    #apparently adding multiple currentindexchanged causes a memory leak unless i sleep, idk why it does this but im kinda dumb
    
    self.ui.VidQualityCombo.currentIndexChanged.connect(lambda: selVidQuality(self))

    # list every model downloaded, and add them to the list
    
    model_filepaths = ([x[0] for x in os.walk(f'{thisdir}/rife-vulkan-models/')])
    models = []
    for model_filepath in model_filepaths:
        if 'rife' in os.path.basename(model_filepath):
            models.append(os.path.basename(model_filepath))
    
    models.sort()
    for model in models:

        
        model = model.replace('r',"R")
        model = model.replace('v','V')
        model = model.replace('a','A')
        self.ui.Rife_Model.addItem(f'{model}')#Adds model to GUI.
        if model == 'Rife-V4.6':
            self.ui.Rife_Model.setCurrentText(f'{model}')