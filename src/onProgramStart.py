from src.settings import *
from src.messages import *
def onApplicationStart(self):
    import os
    
    #Import all modules
    ''' for module in os.listdir(f'{thisdir}/modules/'):
            if module == '__init__.py' or module[-3:] != '.py':
                continue
            __import__(f'modules.{module[:-3]}', locals(), globals())
            
            self.ui.AICombo.addItem(f'{module[:-3]}')'''
    import modules.Rife as rife
    import modules.ESRGAN as esrgan
    settings = Settings()
    from PyQt5.QtGui import QIntValidator, QIcon
    thisdir=os.getcwd()
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
    self.ui.Rife_Times.currentIndexChanged.connect(self.showChangeInFPS)
    self.ui.RifePause.clicked.connect(self.pause_render)
    self.ui.RifeResume.clicked.connect(self.resume_render)
    self.ui.RifeResume.hide()
    self.ui.RifePause.hide()
    self.ui.DiscordRPCBox.stateChanged.connect(lambda: changeDiscordRPC(self))
    if settings.DiscordRPC == 'Enabled':
        self.ui.DiscordRPCBox.setChecked(True)
        
    else:
        self.ui.DiscordRPCBox.setChecked(False)
    os.system('ln -sf {app/com.discordapp.Discord,$XDG_RUNTIME_DIR}/discord-ipc-0') #Enables discord RPC on flatpak
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
    
    #link help buttons
    self.ui.sceneChangeSensativityButton.clicked.connect(lambda: show_scene_change_help(self))
    self.ui.encoderHelpButton.clicked.connect(lambda:  encoder_help(self))
    
    self.ui.RenderPathLabel.setText(f"{settings.RenderDir}")
    self.ui.RenderDirButton.clicked.connect(lambda: selRenderDir(self))
    self.ui.verticalTabWidget.setCurrentWidget(self.ui.verticalTabWidget.findChild(QWidget, 'Rife'))
    self.ui.Input_video_rife.clicked.connect(self.openFileNameDialog)
    self.ui.Output_folder_rife.clicked.connect(self.openFolderDialog)
    self.ui.VideoOptionsFrame.hide()
    self.ui.RenderOptionsFrame.hide()
    self.ui.GeneralOptionsFrame.hide()
   
    
    self.ui.EncoderCombo.currentIndexChanged.connect(lambda: selEncoder(self))
    #apparently adding multiple currentindexchanged causes a memory leak unless i sleep, idk why it does this but im kinda dumb
    
    self.ui.VidQualityCombo.currentIndexChanged.connect(lambda: selVidQuality(self))
    
    # list every model downloaded, and add them to the list
def list_model_downloaded(self):
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