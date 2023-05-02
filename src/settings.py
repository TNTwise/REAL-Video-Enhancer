import os
import csv
thisdir = os.getcwd()
class Settings:
    def __init__(self) -> None:
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
        
    def change_setting(self,setting,svalue):
        original_settings = {}
        with open(f'{thisdir}/files/settings.txt', 'r') as f:
            f = csv.reader(f)
            for row in f:
                original_settings[row[0]] = row[1]
            
            original_settings[setting] = svalue
            os.system(f'rm -rf "{thisdir}/files/settings.txt" && touch "{thisdir}/files/settings.txt"')
            for key,value in original_settings.items():
                with open(f'{thisdir}/files/settings.txt', 'a') as f:
                    f.write(key + ',' + value+'\n')
        self.readSettings()
    def write_defaults(self):
        self.change_setting("Interpolation_Option", f"2X")
        self.change_setting("Rife_Option", f"2.3")
        self.change_setting("IsAnime", "False")