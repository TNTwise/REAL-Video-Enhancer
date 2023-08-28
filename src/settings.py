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
        if ManageFiles.isfile(f"{thisdir}/files/settings.txt") == False:
            ManageFiles.create_file(f"{thisdir}/files/settings.txt")
            self.write_defaults()
        self.write_temp()
        self.readSettings()
    def write_to_settings_file(self,description, option):
    
        with open(f'{thisdir}/files/settings.txt', 'a') as f:
            f.write(description + ","+option + "\n")
    def write_defaults(self):
        with open(f'{thisdir}/files/settings.txt', 'w') as f:
            pass
        self.write_to_settings_file("Image_Type", ".jpg")
        self.write_to_settings_file("rifeversion", "20221029")
        self.write_to_settings_file("esrganversion", "0.2.0")
        self.write_to_settings_file("videoQuality", "18")
        self.write_to_settings_file("Theme", "Dark")
        self.write_to_settings_file("OutputDir", f"{homedir}")
        self.write_to_settings_file("Interpolation_Option", f"2X")
        self.write_to_settings_file("RenderDir" ,f"{thisdir}")
        self.write_to_settings_file('SceneChangeDetection','0.3')
        self.write_to_settings_file('Encoder','264')
        self.write_to_settings_file('ModelDir',f'{thisdir}/models')
        self.write_to_settings_file('RenderType','Optimized')
        self.write_to_settings_file('FrameIncrements', '100')
        self.write_to_settings_file('FrameIncrementsMode', 'Automatic')
        self.write_to_settings_file('DiscordRPC', 'Enabled')
        if HardwareInfo.get_video_memory_linux() == None:
            self.write_to_settings_file('VRAM',f'{HardwareInfo.get_video_memory_linux()}')
        elif  HardwareInfo.get_video_memory_linux() >= 1:
                self.write_to_settings_file('VRAM',f'{HardwareInfo.get_video_memory_linux()}')
        elif HardwareInfo.get_video_memory_linux() < 1:
                self.write_to_settings_file('VRAM','1')
        

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
        try:
            
            self.Image_Type = settings_dict['Image_Type']
        except:
            self.write_to_settings_file("Image_Type", ".jpg")
            self.readSettings()
        try:
            self.videoQuality = settings_dict['videoQuality']
        except:
            self.write_to_settings_file("videoQuality", "18")
            self.readSettings()
        try:
            self.FrameIncrements = int(settings_dict['FrameIncrements']) # need this to be int
        except:
            self.write_to_settings_file('FrameIncrements', '100')
            self.readSettings()
        try:
            self.Theme = settings_dict['Theme'] # need this to be int
        except:
            self.write_to_settings_file('Theme', 'Dark')
            self.readSettings()
        try:
            self.OutputDir = settings_dict['OutputDir']
            if os.path.exists(f'{self.OutputDir}') == False:
                self.write_to_settings_file("OutputDir" ,f"{homedir}")
        except:
            self.write_to_settings_file("OutputDir", f"{homedir}")
            self.readSettings()
        try:
            self.GPUUsage = settings_dict['GPUUsage']
        except:
            self.write_to_settings_file("GPUUsage" ,'Default')
            self.readSettings()
        try:
            self.ModelDir = settings_dict['ModelDir']
        except:
            self.write_to_settings_file("ModelDir" ,f'{thisdir}/models/')
            self.readSettings()
        try:
            self.RenderType = settings_dict['RenderType']
        except:
            self.write_to_settings_file('RenderType','Optimized')
            self.readSettings()
        try:
            self.RenderDir = settings_dict['RenderDir']
            if os.path.exists(f'{self.RenderDir}') == False:
                self.write_to_settings_file("RenderDir" ,f"{thisdir}")
            
        except:
            self.write_to_settings_file("RenderDir" ,f"{thisdir}")
            self.readSettings()
        try:
            self.ExtractionImageType=settings_dict['ExtractionImageType']
        except:
            self.write_to_settings_file("ExtractionImageType" ,"jpg")
            self.readSettings()
        try: 
            self.SceneChangeDetection=settings_dict['SceneChangeDetection']
        except:
            self.write_to_settings_file('SceneChangeDetection','0.3')
            self.readSettings()
        try:
            self.Encoder=settings_dict['Encoder']
        except:
            self.write_to_settings_file('Encoder','264')
            self.readSettings()
        try:
            self.DiscordRPC=settings_dict['DiscordRPC']
            if self.DiscordRPC == 'Enabled':
                self.DiscordRPC == True
            else:
                self.DiscordRPC == False
        except:
            self.write_to_settings_file('DiscordRPC', 'Enabled')
            self.readSettings()       
        try:
            self.ModelDir=settings_dict['ModelDir']
        except:
            self.write_to_settings_file('ModelDir',f'{thisdir}/models')
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
        try:
            self.FrameIncrementsMode = settings_dict['FrameIncrementsMode']
        except:
            self.write_to_settings_file('FrameIncrementsMode', 'Automatic')
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
    def write_temp(self):
        self.change_setting("Interpolation_Option", f"2X")
        self.change_setting("Rife_Option", f"2.3")
        self.change_setting("IsAnime", "False")

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
    if 'Optimized' in self.ui.renderTypeCombo.currentText():
        settings.change_setting('RenderType','Optimized')
        
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
        self.ui.frameIncrementHelp.hide()
        self.ui.frameIncrementSpinBox.hide()
        self.ui.label_7.hide()
    else:
        self.ui.frameIncrementHelp.show()
        self.ui.frameIncrementSpinBox.show()
        self.ui.label_7.show()