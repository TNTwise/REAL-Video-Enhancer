import os
import csv
thisdir = os.getcwd()
from src.return_data import *
homedir = os.path.expanduser(r"~")
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QMessageBox

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
        self.write_to_settings_file("Image_Type", "png")
        self.write_to_settings_file("IsAnime", "False")
        self.write_to_settings_file("Repository", "stable")
        self.write_to_settings_file("rifeversion", "20221029")
        self.write_to_settings_file("esrganversion", "0.2.0")
        self.write_to_settings_file("videoQuality", "18")
        self.write_to_settings_file("Theme", "Dark")
        self.write_to_settings_file("OutputDir", f"{homedir}")
        self.write_to_settings_file("Interpolation_Option", f"2X")
        self.write_to_settings_file("Rife_Option" ,'2.3')
        self.write_to_settings_file("GPUUsage" ,'Default')
        self.write_to_settings_file("RenderDevice" ,'GPU')
        self.write_to_settings_file("RenderDir" ,f"{thisdir}")
        self.write_to_settings_file("ExtractionImageType" ,"jpg")
        self.write_to_settings_file('SceneChangeDetection','0.3')
        self.write_to_settings_file('Encoder','264')

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
            self.write_to_settings_file("Image_Type", "png")
            self.readSettings()
        try:
            self.videoQuality = settings_dict['videoQuality']
        except:
            self.write_to_settings_file("videoQuality", "18")
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
            self.write_to_settings_file('DiscordRPC', 'Disabled')
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
    self.render_folder = QFileDialog.getExistingDirectory(self, 'Open Folder')
    settings.change_setting("RenderDir",f"{self.render_folder}")
    
    self.ui.RenderPathLabel.setText(f"{settings.RenderDir}")

def selEncoder(self):
    settings = Settings()
    if '.264' in self.ui.EncoderCombo.currentText():
        
        settings.change_setting('Encoder','264')
    if '.265' in self.ui.EncoderCombo.currentText():
        settings.change_setting('Encoder','265')
    self.encoder = settings.Encoder

def selVidQuality(self):
    settings = Settings()
    if self.ui.VidQualityCombo.currentText() == 'Lossless':
        settings.change_setting('videoQuality', '10')
    if self.ui.VidQualityCombo.currentText() == 'High':
        settings.change_setting('videoQuality', '14')
    if self.ui.VidQualityCombo.currentText() == 'Medium':
        settings.change_setting('videoQuality', '18')
    if self.ui.VidQualityCombo.currentText() == 'Low':
        settings.change_setting('videoQuality', '22')
    self.videoQuality = settings.videoQuality