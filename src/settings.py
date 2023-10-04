import os
import csv
import src.thisdir
thisdir = src.thisdir.thisdir()
from src.return_data import *
homedir = os.path.expanduser(r"~")
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QMessageBox
#im going to eventually redo this
class Settings:
    def __init__(self) -> None:
        
        ManageFiles.create_folder(f"{thisdir}/files")
        ManageFiles.create_file(f"{thisdir}/files/settings.txt")
        self.readSettings()
    def write_to_settings_file(self,description, option):
    
        with open(f'{thisdir}/files/settings.txt', 'a') as f:
            f.write(description + ","+option + "\n")
    
    def check_and_write_setting(self,setting_match,value,settings_dict):
        setting,setting_identifier = setting_match
        try:
            setting = settings_dict[value]
            print(settings_dict)
            print(setting)
        except:
            self.write_to_settings_file(setting_identifier,value)
            self.readSettings()
    def readSettings(self):
        settings_dict = {}
        with open(f'{thisdir}/files/settings.txt', 'r') as f:
            f = csv.reader(f)
            for row in f:
                try:
                    settings_dict[row[0]] = row[1]
                except:
                    pass
            
        default_settings = {
    "FixFFMpegCatchup": "Disabled",
    "Image_Type": ".jpg",
    "videoQuality": "18",
    "FrameIncrements": '100',
    "Theme": 'Dark',
    "OutputDir": f"{homedir}",
    "GPUUsage": 'Default',
    "ModelDir": f'{thisdir}/models/',
    "RenderType": 'Optimized',
    "SceneChangeDetectionMode": 'Enabled',
    "RenderDir": f"{thisdir}",
    "ExtractionImageType": "jpg",
    "SceneChangeDetection": '0.3',
    "Encoder": '264',
    "DiscordRPC": 'Enabled',
    "ModelDir": f'{thisdir}/models/',
    'FrameIncrementsMode': 'Automatic',
    'UpdateChannel': 'Stable',
    'Version': '20231004',
}

        for setting, default_value in default_settings.items():
            try:
                setattr(self, setting, settings_dict[setting])
                if setting in ["OutputDir", "RenderDir"] and not os.path.exists(getattr(self, setting)):
                    raise Exception
                
                if setting == "FrameIncrements":
                    self.FrameIncrements = int(self.FrameIncrements)
            except:
                self.write_to_settings_file(setting, default_value)
                self.readSettings()
        try:
            self.VRAM = settings_dict['VRAM']
        except:
            if HardwareInfo.get_video_memory_linux() == None:
                self.write_to_settings_file('VRAM',f'{HardwareInfo.get_video_memory_linux()}')
            elif  HardwareInfo.get_video_memory_linux() >= 1:
                self.write_to_settings_file('VRAM',f'{HardwareInfo.get_video_memory_linux()}')
            elif HardwareInfo.get_video_memory_linux() < 1:
                self.write_to_settings_file('VRAM','1')
            self.readSettings()
        
        
    def change_setting(self,setting,svalue):
        original_settings = {}
        with open(f'{thisdir}/files/settings.txt', 'r') as f:
            f = csv.reader(f)
            for row in f:
                original_settings[row[0]] = row[1]
            
            original_settings[setting] = svalue
            os.remove(f"{thisdir}/files/settings.txt")
            for key,value in original_settings.items():
                with open(f'{thisdir}/files/settings.txt', 'a') as f:
                    f.write(key + ',' + value+'\n')
        self.readSettings()
    def is_setting(setting):
        
        with open(f'{thisdir}/files/settings.txt', 'r') as f:
            f = csv.reader(f)
            for row in f:
                if row[0] == setting:
                    return True
            return False
   

def changeDiscordRPC(self):
    settings = Settings()
    if self.ui.DiscordRPCBox.isChecked() == True:
        settings.change_setting('DiscordRPC',f'Enabled')
    else:
        settings.change_setting('DiscordRPC',f'Disabled')


def changeSceneDetection(self):
        settings = Settings()
        if len(self.ui.sceneChangeLineEdit.text()) > 0 and int(self.ui.sceneChangeLineEdit.text()) != 0:
            settings.change_setting('SceneChangeDetection', f'0.{self.ui.sceneChangeLineEdit.text()}')
def selRenderDir(self):
    settings = Settings()
    render_folder = QFileDialog.getExistingDirectory(self, 'Open Folder')
    if render_folder != '':
        settings.change_setting("RenderDir",f"{self.render_folder}")
        
        self.ui.RenderPathLabel.setText(f"{settings.RenderDir}")
def selOutputDir(self):
    settings = Settings()
    output_folder = QFileDialog.getExistingDirectory(self, 'Open Folder')
    if output_folder != '':
        self.output_folder = output_folder
        settings.change_setting("OutputDir",f"{self.output_folder}")
        
        self.ui.OutputDirectoryLabel.setText(f"{settings.OutputDir}")
    
def selEncoder(self):
    settings = Settings()
    if '.264' in self.ui.EncoderCombo.currentText():
        
        settings.change_setting('Encoder','264')
    if '.265' in self.ui.EncoderCombo.currentText():
        settings.change_setting('Encoder','265')
    self.encoder = settings.Encoder

def selRenderType(self):
    settings = Settings()
    if 'Classic' in self.ui.renderTypeCombo.currentText():
        
        settings.change_setting('RenderType','Classic')
    if 'Optimized' == self.ui.renderTypeCombo.currentText():
        settings.change_setting('RenderType','Optimized')
    if 'Optimized (Incremental)' == self.ui.renderTypeCombo.currentText():
        settings.change_setting('RenderType','Optimized (Incremental)')
        
def selFrameIncrements(value):
    settings = Settings()
    settings.change_setting('FrameIncrements',f'{value}')
    
def selVidQuality(self):
    settings = Settings()
    if self.ui.VidQualityCombo.currentText() == 'Lossless':
        settings.change_setting('videoQuality', '10')
    if self.ui.VidQualityCombo.currentText() == 'Very High':
        settings.change_setting('videoQuality', '14')
    if self.ui.VidQualityCombo.currentText() == 'High':
         settings.change_setting('videoQuality', '18')
    if self.ui.VidQualityCombo.currentText() == 'Medium':
       settings.change_setting('videoQuality', '20')
    if self.ui.VidQualityCombo.currentText() == 'Low':
        settings.change_setting('videoQuality', '22')
    self.videoQuality = settings.videoQuality
    

def selFrameIncrementsMode(self):
    settings = Settings()
    settings.change_setting('FrameIncrementsMode',self.ui.frameIncrementsModeCombo.currentText())
    if settings.FrameIncrementsMode == 'Automatic':
        self.ui.frameIncrementSpinBox.hide()
        self.ui.label_7.hide()
    else:
        self.ui.frameIncrementSpinBox.show()
        self.ui.label_7.show()
def selSceneDetectionMode(self):
    
    settings = Settings()
    if self.ui.sceneChangeDetectionCheckBox.isChecked() == True:
        settings.change_setting('SceneChangeDetectionMode','Enabled')
        self.ui.label_3.show()
        self.ui.sceneChangeSensativityButton.show()
        self.ui.sceneChangeLineEdit.show()
    else:
        settings.change_setting('SceneChangeDetectionMode','Disabled')
        self.ui.label_3.hide()
        self.ui.sceneChangeSensativityButton.hide()
        self.ui.sceneChangeLineEdit.hide()