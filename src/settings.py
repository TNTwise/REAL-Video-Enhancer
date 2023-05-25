import os
import csv
thisdir = os.getcwd()
from src.return_data import *
homedir = os.path.expanduser(r"~")

class Settings:
    def __init__(self) -> None:

        ManageFiles.create_folder(f"{thisdir}/files")
        if ManageFiles.isfile(f"{thisdir}/files/settings.txt") == False:
            ManageFiles.create_file(f"{thisdir}/files/settings.txt")
            self.write_defaults()
        self.write_temp()
        self.readSettings()
    def write_to_settings_file(description, option):
    
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
            self.Rife_Option = settings_dict['Rife_Option']
            self.Interpolation_Option = settings_dict['Interpolation_Option']
            self.Repository = settings_dict['Repository']
            self.Image_Type = settings_dict['Image_Type']
            self.IsAnime = settings_dict['IsAnime']
            self.rifeversion = settings_dict['rifeversion']
            self.esrganversion = settings_dict['esrganversion']
            self.videoQuality = settings_dict['videoQuality']
            self.Theme = settings_dict['Theme']
            self.OutputDir = settings_dict['OutputDir']
            self.GPUUsage = settings_dict['GPUUsage']
            self.RenderDevice = settings_dict['RenderDevice']
            self.RenderDir = settings_dict['RenderDir']
            self.ExtractionImageType=settings_dict['ExtractionImageType']
            self.SceneChangeDetection=settings_dict['SceneChangeDetection']
            self.Encoder=settings_dict['Encoder']
        except:
            self.write_defaults()
        
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
