from src.settings import *
from src.messages import *
import src.queue.queue as queue
import multiprocessing
import src.checks as checks
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
import os
from src.log import *
from src.getLinkVideo.get_video import *
def open_link():
        url = QUrl(r'https://github.com/upscayl/upscayl/wiki/%F0%9F%96%A5%EF%B8%8F-Model-Conversion%E2%80%89%E2%80%93%E2%80%89Get-more-models!')
        QDesktopServices.openUrl(url)

def bindButtons(self):
    settings = Settings()
    self.ui.RifeSettings.clicked.connect(lambda: src.getModels.get_models_settings.get_rife(self))
    self.ui.esrganHelpModel.clicked.connect(open_link)
    self.ui.Rife_Times.currentIndexChanged.connect(self.showChangeInFPS)
    self.ui.vramAmountSpinbox.valueChanged.connect(self.changeVRAM)
    self.ui.AICombo.currentIndexChanged.connect(self.switchUI)
    self.ui.modeCombo.currentIndexChanged.connect(self.switchMode)
    self.ui.DiscordRPCBox.stateChanged.connect(lambda: changeDiscordRPC(self))
    self.ui.sceneChangeLineEdit.textChanged.connect(lambda: changeSceneDetection(self))
    self.ui.frameIncrementsModeCombo.currentTextChanged.connect(lambda: selFrameIncrementsMode(self))
    self.ui.sceneChangeDetectionCheckBox.stateChanged.connect(lambda: selSceneDetectionMode(self))
    self.ui.RenderDirButton.clicked.connect(lambda: selRenderDir(self))
    
    self.ui.Input_video_rife.clicked.connect(lambda: self.openFileNameDialog('Video',['.mp4','.mkv','.webm','.mov','.webm','avi']))
    
    
    self.ui.Output_folder_rife.clicked.connect(self.openFolderDialog)
    self.ui.frameIncrementSpinBox.valueChanged.connect(lambda: selFrameIncrements(self.ui.frameIncrementSpinBox.value()))
    #link buttons
    self.ui.frameIncrementHelp.clicked.connect(lambda: frame_increments_help(self))
    self.ui.SettingsMenus.clicked.connect(self.settings_menu)
    self.ui.OutputDirectoryButton.clicked.connect(lambda: selOutputDir(self))
    self.ui.resetSettingsButton.clicked.connect(self.restore_default_settings)
    self.ui.vramAmountHelpButton.clicked.connect(lambda: vram_help(self))
    self.ui.logButton.clicked.connect(lambda: viewLogs(self))
    self.ui.RifePause.clicked.connect(self.pause_render)
    self.ui.RifeResume.clicked.connect(self.resume_render)
    self.ui.sceneChangeSensativityButton.clicked.connect(lambda: show_scene_change_help(self))
    self.ui.encoderHelpButton.clicked.connect(lambda:  encoder_help(self))
    self.ui.ensembleHelpButton.clicked.connect(lambda: ensemble_help(self))
    self.ui.renderTypeCombo.currentIndexChanged.connect(lambda: selRenderType(self))
    self.ui.renderTypeHelpButton.clicked.connect(lambda: render_help(self))
    self.ui.Rife_Model.currentIndexChanged.connect(self.greyOutRifeTimes)
    self.ui.imageHelpButton.clicked.connect(lambda: image_help(self))
    self.ui.imageComboBox.currentIndexChanged.connect(lambda: settings.change_setting('Image_Type', f'{self.ui.imageComboBox.currentText()}'))
    self.ui.EncoderCombo.currentIndexChanged.connect(lambda: selEncoder(self))
    #apparently adding multiple currentindexchanged causes a memory leak unless i sleep, idk why it does this but im kinda dumb
    self.ui.ESRGANModelSelectButton.clicked.connect(lambda: self.openFileNameDialog('Model',['.bin']))
    self.ui.Input_video_rife_url.clicked.connect(lambda: get_linked_video(self))
    self.ui.VidQualityCombo.currentIndexChanged.connect(lambda: selVidQuality(self))
    self.ui.QueueButton.clicked.connect(lambda: queue.addToQueue(self))
    self.ui.defaultRifeModel.currentIndexChanged.connect(lambda: settings.change_setting('DefaultRifeModel',f'{self.ui.defaultRifeModel.currentText()}'))
    self.ui.InstallButton.clicked.connect(lambda: src.getModels.get_models_settings.run_install_models_from_settings(self))
    self.ui.NotificationsCheckBox.stateChanged.connect(lambda: Notifications(self))
    self.ui.uninstallButton.clicked.connect(lambda: uninstallAPP(self))
    self.ui.sceneChangeSensativityButton.setIcon(QIcon(f"{thisdir}/icons/Rife-ESRGAN-Video-Settings - Help.png"))
    self.ui.encoderHelpButton.setIcon(QIcon(f"{thisdir}/icons/Rife-ESRGAN-Video-Settings - Help.png"))
    self.ui.imageHelpButton.setIcon(QIcon(f"{thisdir}/icons/Rife-ESRGAN-Video-Settings - Help.png"))
    self.ui.vramAmountHelpButton.setIcon(QIcon(f"{thisdir}/icons/Rife-ESRGAN-Video-Settings - Help.png"))
    self.ui.renderTypeHelpButton.setIcon(QIcon(f"{thisdir}/icons/Rife-ESRGAN-Video-Settings - Help.png"))
    self.ui.frameIncrementHelp.setIcon(QIcon(f"{thisdir}/icons/Rife-ESRGAN-Video-Settings - Help.png"))
    self.ui.esrganHelpModel.setIcon(QIcon(f"{thisdir}/icons/Rife-ESRGAN-Video-Settings - Help.png"))
    self.ui.ensembleHelpButton.setIcon(QIcon(f"{thisdir}/icons/Rife-ESRGAN-Video-Settings - Help.png"))
def onApplicationStart(self):
    
    #this is kind of a mess
    import modules.Rife as rife
    import modules.ESRGAN as esrgan
    settings = Settings()
    from PyQt5.QtGui import QIntValidator, QIcon
    import src.thisdir
    thisdir = src.thisdir.thisdir()
    self.ui.AICombo.clear() # needs to be in this order, before SwitchUI is called
    set_model_params(self)
    #get esrgan models
    if not (self.ui.RealESRGANCheckBox.isChecked()):
        self.ui.ESRGANModelSelectButton.hide()
        self.ui.label_20.hide()
        self.ui.esrganHelpModel.hide()
    else:
        self.ui.ESRGANModelSelectButton.show()
        self.ui.label_20.show()
        self.ui.esrganHelpModel.show()
    self.input_file = ''
    
    self.setWindowIcon(QIcon(f'{thisdir}/icons/logo v1.png'))
    
    self.settings = Settings()
    self.gpuMemory=self.settings.VRAM
    
    self.switchUI()
    if self.gpuMemory != 'None':
        
        self.ui.vramAmountSpinbox.setValue(int(self.gpuMemory))
        
    else:
        if self.settings.VRAM == 'None':
            cannot_detect_vram(self)
        self.ui.vramAmountSpinbox.setValue(1)
    
    
    
    
    
    self.ui.installModelsProgressBar.setMaximum(100)
        
    self.ui.vramAmountSpinbox.setMinimum(1)
    
    #Define Variables
    self.input_file = ''
    self.output_folder = ''
    self.videoQuality = settings.videoQuality
    self.encoder = settings.Encoder
    if os.path.exists(f"{settings.RenderDir}") == False:
        settings.change_setting('RenderDir',f'{thisdir}')
    
    
    
    self.ui.InstallModelsFrame.hide()
    self.ui.RifeResume.hide()
    self.ui.RifePause.hide()
    
    
    if settings.DiscordRPC == 'Enabled':
        self.ui.DiscordRPCBox.setChecked(True)
        
    else:
        self.ui.DiscordRPCBox.setChecked(False)
    os.system('ln -sf {app/com.discordapp.Discord,$XDG_RUNTIME_DIR}/discord-ipc-0') #Enables discord RPC on flatpak
    onlyInt = QIntValidator()
    onlyInt.setRange(0, 9)
    self.ui.sceneChangeLineEdit.setValidator(onlyInt)
    
    self.ui.sceneChangeLineEdit.setText(settings.SceneChangeDetection[2])
    if self.encoder == '264':
        self.ui.EncoderCombo.setCurrentIndex(0)
    if self.encoder == '265':
        self.ui.EncoderCombo.setCurrentIndex(1)
    if self.videoQuality == '10':
        self.ui.VidQualityCombo.setCurrentText('Lossless')
    if self.videoQuality == '14':
        self.ui.VidQualityCombo.setCurrentText('Very High')
    if self.videoQuality == '18':
        self.ui.VidQualityCombo.setCurrentText('High')
    if self.videoQuality == '20':
        self.ui.VidQualityCombo.setCurrentText('Medium')
    if self.videoQuality == '22':
        self.ui.VidQualityCombo.setCurrentText('Low')
    if self.settings.RenderType == 'Classic':
        self.ui.renderTypeCombo.setCurrentIndex(0)
    elif self.settings.RenderType == 'Optimized':
        self.ui.renderTypeCombo.setCurrentIndex(1)
    else:
        self.ui.renderTypeCombo.setCurrentIndex(2)
    
    if settings.SceneChangeDetectionMode == 'Enabled':
        self.ui.sceneChangeDetectionCheckBox.setChecked(True)
        self.ui.label_3.show()
        self.ui.sceneChangeSensativityButton.show()
        self.ui.sceneChangeLineEdit.show()
    else:
        self.ui.sceneChangeDetectionCheckBox.setChecked(False)
        self.ui.label_3.hide()
        self.ui.sceneChangeSensativityButton.hide()
        self.ui.sceneChangeLineEdit.hide()
    
    self.ui.OutputDirectoryLabel.setText(settings.OutputDir)
    self.ui.frameIncrementSpinBox.setValue(int(settings.FrameIncrements))
    
            
    self.ui.RenderPathLabel.setText(f"{settings.RenderDir}")
    if settings.Notifications == 'Enabled':
        self.ui.NotificationsCheckBox.setChecked(True)
    else:
        self.ui.NotificationsCheckBox.setChecked(False)
    self.ui.VideoOptionsFrame.hide()
    self.ui.RenderOptionsFrame.hide()
    self.ui.GeneralOptionsFrame.hide()
    
    
    
    self.ui.QueueButton.hide()
    self.ui.QueueListWidget.hide()
    self.QueueList=[]
    self.setDirectories()
    self.ui.imageComboBox.setCurrentText(f'{settings.Image_Type}')
    
    # list every model downloaded, and add them to the list
    self.ui.SettingsMenus.setCurrentRow(0)
    self.ui.GeneralOptionsFrame.show()

    #set default model in settings
    self.ui.defaultRifeModel.setCurrentText(f'{settings.DefaultRifeModel}')
    
def set_model_params(self):
    models_installed = checks.check_for_individual_models()
    settings = Settings()
    self.model_labels = {}
    self.ui.RifeSettings.setEnabled(False)
    for i in models_installed:
                
                if 'Rife' == i:
                    self.ui.RifeCheckBox.setChecked(True)
                    self.model_labels['Rife'] = 'interpolation'
                    self.ui.RifeSettings.setEnabled(True)
                if 'RealESRGAN' == i:
                    self.ui.RealESRGANCheckBox.setChecked(True)
                    self.model_labels['RealESRGAN'] = 'upscaling'
                if 'RealCUGAN' == i:
                    self.ui.RealCUGANCheckBox.setChecked(True)
                    self.model_labels['RealCUGAN'] = 'upscaling'
                if 'RealSR' == i:
                    self.ui.RealSRCheckBox.setChecked(True)
                    self.model_labels['RealSR'] = 'upscaling'
                if 'Waifu2X' == i:
                    self.ui.Waifu2xCheckBox.setChecked(True)
                    self.model_labels['Waifu2X'] = 'upscaling'
                if 'IFRNET' == i:
                    self.ui.CainCheckBox.setChecked(True)
                    self.model_labels['IFRNET'] = 'interpolation'
    #not efficient but im lazy so cry abt it
    self.ui.modeCombo.clear()
    upscale_list=[]
    for i in range(self.ui.modeCombo.count()):
            item_text = self.ui.modeCombo.itemText(i)
            upscale_list.append(item_text)
    if 'Interpolation' not in upscale_list:

        for key,value in self.model_labels.items():
            if value == 'interpolation':
                self.ui.modeCombo.addItem('Interpolation')
                break
    if 'Upscaling' not in upscale_list:
        for key,value in self.model_labels.items():
            if value == 'upscaling':
                self.ui.modeCombo.addItem('Upscaling')
                break

    self.switchMode()