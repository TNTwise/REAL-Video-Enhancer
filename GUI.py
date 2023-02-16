import PyQt5.QtWidgets as qtw
import PyQt5.QtGui as qtg
import os
from glob import glob
import re
import csv
thisdir = os.getcwd()
homedir = os.path.expanduser(r"~")
class getModels:
    def __init__():
        super().__init__()


class settings_file:
    def __init__(self):
        if os.path.exists(f"{thisdir}/files/") == False:
            os.mkdir(f"{thisdir}/files/")
        if os.path.isfile(f"{thisdir}/files/settings.txt") == False:
            os.mknod(f"{thisdir}/files/settings.txt")
            self.write_defaults()
        #Start default settings processces
                

    def write_to_settings_file(self, description, option):
        with open(f'{thisdir}/files/settings.txt', 'a') as f:
            f.write(description + ","+option + "\n")
    def read_settings(self):
        global settings_dict
        settings_dict = {}

        with open(f'{thisdir}/files/settings.txt', 'r') as f:
            f = csv.reader(f)
            for row in f:
                try:
                    settings_dict[row[0]] = row[1]
                except:
                    pass
        try:
            global Rife_Option
            Rife_Option = settings_dict['Rife_Option']
            global Interpolation_Option
            Interpolation_Option = settings_dict['Interpolation_Option']
            global Repository
            Repository = settings_dict['Repository']
            global Image_Type
            Image_Type = settings_dict['Image_Type']
            global IsAnime
            IsAnime = settings_dict['IsAnime']
            global rifeversion
            rifeversion = settings_dict['rifeversion']
            global esrganversion
            esrganversion = settings_dict['esrganversion']
            global videoQuality
            videoQuality = settings_dict['videoQuality']
            global Theme
            Theme = settings_dict['Theme']
            global OutputDir
            OutputDir = settings_dict['OutputDir']
            global GPUUsage
            GPUUsage = settings_dict['GPUUsage']
            global RenderDevice
            RenderDevice = settings_dict['RenderDevice']
            global RenderDir
            RenderDir = settings_dict['RenderDir']
        except:
            os.system(f'rm -rf "{thisdir}/files/settings.txt"')
            os.mknod(f'{thisdir}/files/settings.txt')
            self.write_to_settings_file("Image_Type", "webp")
            self.write_to_settings_file("IsAnime", "False")
            self.write_to_settings_file("Repository", "stable")
            self.write_to_settings_file("rifeversion", "20221029")
            self.write_to_settings_file("esrganversion", "0.2.0")
            self.write_to_settings_file("videoQuality", "14")
            self.write_to_settings_file("Theme", "Light")
            self.write_to_settings_file("OutputDir", f"{homedir}")
            self.write_to_settings_file("Interpolation_Option", f"2X")
            self.write_to_settings_file("Rife_Option" ,'2.3')
            self.write_to_settings_file("GPUUsage" ,'Default')
            self.write_to_settings_file("RenderDevice" ,'GPU')
            self.write_to_settings_file("RenderDir" ,f"{thisdir}")
            settings_dict = {}
            with open(f'{thisdir}/files/settings.txt', 'r') as f:
                f = csv.reader(f)
                for row in f:
                    settings_dict[row[0]] = row[1]
            Rife_Option = settings_dict['Rife_Option']
        
            Interpolation_Option = settings_dict['Interpolation_Option']
        
            Repository = settings_dict['Repository']
        
            Image_Type = settings_dict['Image_Type']
        
            IsAnime = settings_dict['IsAnime']
            rifeversion = settings_dict['rifeversion']
            esrganversion = settings_dict['esrganversion']
            videoQuality = settings_dict['videoQuality']
            Theme = settings_dict['Theme']
            OutputDir = settings_dict['OutputDir']
            GPUUsage = settings_dict['GPUUsage']
            RenderDevice = settings_dict['RenderDevice']
            RenderDir = settings_dict['RenderDir']
    def write_defaults(self):
        self.write_to_settings_file("Image_Type", "webp")
        self.write_to_settings_file("IsAnime", "False")
        self.rite_to_settings_file("Repository", "stable")
        self.write_to_settings_file("rifeversion", "20221029")
        self.write_to_settings_file("esrganversion", "0.2.0")
        self.write_to_settings_file("videoQuality", "14")
        self.write_to_settings_file("Theme", "Light")
        self.write_to_settings_file("OutputDir", f"{homedir}")
        self.write_to_settings_file("Interpolation_Option", f"2X")
        self.write_to_settings_file("Rife_Option" ,'2.3')
        self.write_to_settings_file("GPUUsage" ,'Default')
        self.write_to_settings_file("RenderDevice" ,'GPU')
        self.write_to_settings_file("RenderDir" ,f"{thisdir}")
        try:
            read_settings()
        except:
            pass
    def change_settings(self,setting,svalue):
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
        self.read_settings()


class MainWindow(qtw.QWidget):
    def __init__(self):
        super().__init__()
        # Sets window title
        self.setWindowTitle("Rife - ESRGAN - App - Linux")
        
        self.setLayout(qtw.QVBoxLayout())
        self.layout_rife()
        #Show the app
        self.show()
        
    def layout_rife(self):
        my_label = qtw.QLabel("Pick rife ver")
        self.layout().addWidget(my_label)
        
        # Change font size

        my_label.setFont(qtg.QFont("Helvetica", 18))
        my_combo = qtw.QComboBox(self)
        os.chdir(f"{thisdir}/rife-vulkan-models/")
        rife_models = glob("rife-*")
        os.chdir(f"{thisdir}")
        my_combo.addItem('rife')
        for i in rife_models:
            if i != "rife-ncnn-vulkan":
                my_combo.addItem(i)
        
        self.layout().addWidget(my_combo)
        my_combo.activated[str].connect(self.onChanged)    
    def onChanged(self,text):
        change_setting = settings_file()
        change_setting.change_settings("Rife_Option", f"{text}")
#start the app

app = qtw.QApplication([])
mw = MainWindow()

app.exec_()
