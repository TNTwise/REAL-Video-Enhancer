# WORK IN PROGRESS!!!!

import os
import sys
import requests
import re
from zipfile import ZipFile
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QIcon
import src.messages
import src.getModels.SelectModels as SelectModels
import src.getModels.Download as DownloadUI
global rife_install_list
from PyQt5.QtCore import QThread, pyqtSignal, QObject, pyqtSlot
from src.messages import *
from src.checks import *
import tarfile
from sys import exit
from src.getModels.rifeModelsFunctions import *
from src.settings import *
settings = Settings()

import src.thisdir
from src.log import log
thisdir = src.thisdir.thisdir()
import src.getModels.SelectAI as SelectAI
import traceback
class Worker(QObject):
    finished = pyqtSignal()
    intReady = pyqtSignal(list)
    def __init__(self,parent):
          self.main=parent
          QThread.__init__(self,None)
    @pyqtSlot()
    
    def install_modules(self):
            
            
            settings = Settings()
            try:
                    os.system(f'touch "{thisdir}/models.txt"')
                    
                    
                    
                    total_size_in_bytes=0
                    data_downloaded=0
                    for link,name in install_modules_dict.items():
                        response = requests.get(link, stream=True)
                        total_size_in_bytes+= int(response.headers.get('content-length', 0))
                    if check_if_enough_space_for_install(total_size_in_bytes) == False:
                         return 0
                    for link,name in install_modules_dict.items():
                        response = requests.get(link, stream=True)
                        with open(f'{thisdir}/files/{name}', 'wb') as f:
                                for data in response.iter_content(1024):
                                    f.write(data)
                                    data_downloaded+=1024
                                    self.intReady.emit([int(data_downloaded),total_size_in_bytes]) # sends back data to main thread# sends back data to main thread
                    if os.path.exists(f"{settings.ModelDir}") == False:
                        os.mkdir(f"{settings.ModelDir}")
                        os.mkdir(f"{settings.ModelDir}/rife")
                    for i in os.listdir(f'{thisdir}/files/'):
                        if os.path.exists(f'{thisdir}/bin/') == False:
                            os.mkdir(f'{thisdir}/bin/')
                        if i == 'ffmpeg':
                             os.system(f'chmod +x "{thisdir}/files/ffmpeg"')
                             os.system(f'mv "{thisdir}/files/ffmpeg" "{thisdir}/bin/"')
                        if i == 'yt-dlp_linux':
                             os.system(f'chmod +x "{thisdir}/files/yt-dlp_linux"')
                             os.system(f'mv "{thisdir}/files/yt-dlp_linux" "{thisdir}/bin/"')
                        if i == 'glxinfo':
                             os.system(f'chmod +x "{thisdir}/files/glxinfo"')
                             os.system(f'mv "{thisdir}/files/glxinfo" "{thisdir}/bin/"')
                        
                    for i in os.listdir(f'{thisdir}/files/'):
                        
                             
                        if '.zip' in i:

                            with ZipFile(f'{thisdir}/files/{i}', 'r') as zip_ref:
                                name=i.replace('.zip','')
                                original_ai_name_ncnn_vulkan = re.findall(r'[\w]*-ncnn-vulkan', name)[0]
                                original_ai_name = original_ai_name_ncnn_vulkan.replace('-ncnn-vulkan','')


                                zip_ref.extractall(f'{thisdir}/files/')

                            os.system(f'mv "{thisdir}/files/{name}" "{settings.ModelDir}/{original_ai_name}"')
                            os.system(f'chmod +x "{settings.ModelDir}/{original_ai_name}/{original_ai_name_ncnn_vulkan}"')

                        if '.tar.gz' in i:
                            with tarfile.open(f'{thisdir}/files/{i}','r') as f:
                                f.extractall(f'{settings.ModelDir}/rife/')


                    os.system(f'mv "{thisdir}/files/rife-ncnn-vulkan" "{settings.ModelDir}/rife"')
                    os.system(f'chmod +x "{settings.ModelDir}/rife/rife-ncnn-vulkan"')
                    for i in os.listdir(f'{thisdir}/files/'):
                         if '.txt' not in i:
                              os.remove(f'{thisdir}/files/{i}')
                    self.finished.emit()
            except Exception as e:
                traceback_info = traceback.format_exc()
                log(f'{e} {traceback_info}')
                self.main.showDialogBox(e)
                
                
def clear_files():
     for i in os.listdir(f'{thisdir}/files/'):
                         if '.txt' not in i:
                              try:
                                os.remove(f'{thisdir}/files/{i}')
                              except:
                                os.system(f'rm -rf "{thisdir}/files/{i}"')
def install_icons(self):
                if os.path.exists(f'{thisdir}/icons/') == False:
                    if check_if_online():
                        try:
                            print('Downloaded Icons')
                            url = 'https://github.com/TNTwise/REAL-Video-Enhancer/raw/main/github/icons.zip'
                            local_filename = url.split('/')[-1]
                            r = requests.get(url)
                            f = open(f'{thisdir}/{local_filename}', 'wb')
                            for chunk in r.iter_content(chunk_size=512 * 1024): 
                                if chunk: # filter out keep-alive new chunks
                                    f.write(chunk)
                            f.close()
                            with ZipFile(f'{thisdir}/{local_filename}','r') as f:
                                f.extractall(path=f'{thisdir}/')
                            os.remove(f'{thisdir}/{local_filename}')
                        except:
                             failed_download(self)
                    else:
                        failed_download(self)
                os.chdir(f'{thisdir}')
def choose_models(self):
    
    global window
    window = ChooseModels(self)
    window.show()

if check_for_individual_models() == None or check_for_each_binary() == False:
    class ChooseAI(QtWidgets.QMainWindow):
            def __init__(self):
                super(ChooseAI, self).__init__()
                self.ui = SelectAI.Ui_MainWindow()
                install_icons(self)
                self.ui.setupUi(self)
                self.pinFunctions()
            
                self.show()
            def showDialogBox(self,message,displayInfoIcon=False):
                icon = QIcon(f"{thisdir}/icons/Rife-ESRGAN-Video-Settings - Info.png")
                msg = QMessageBox()
                msg.setWindowTitle(" ")
                if displayInfoIcon == True:
                    msg.setIconPixmap(icon.pixmap(32, 32))
                msg.setText(f"{message}")

                msg.exec_()
            def pinFunctions(self):
                 self.ui.InstallButton.clicked.connect(self.next)
                 self.ui.RifeSettings.clicked.connect(lambda: choose_models(self))
                 
            def next(self):
                global install_modules_dict
                install_modules_dict={}
                global rife_install_list
                rife_install_list=[]
                try:
                    with open(f'{thisdir}/models.txt', 'r') as f:
                            for i in f.readlines():
                                print(i)
                                i=i.replace('\n','')
                                rife_install_list.append(i)
                except:
                     rife_install_list.append('rife-v4.6')
                '''https://github.com/nihui/realcugan-ncnn-vulkan/releases/download/20220728/realcugan-ncnn-vulkan-20220728-ubuntu.zip':'realcugan-ncnn-vulkan-20220728-ubuntu.zip',
                'https://github.com/nihui/cain-ncnn-vulkan/releases/download/20220728/cain-ncnn-vulkan-20220728-ubuntu.zip':'cain-ncnn-vulkan-20220728-ubuntu.zip',
                '''
                install_modules_dict = {'https://raw.githubusercontent.com/TNTwise/REAL-Video-Enhancer/main/bin/ffmpeg':'ffmpeg',
'https://raw.githubusercontent.com/TNTwise/REAL-Video-Enhancer/main/bin/yt-dlp_linux':'yt-dlp_linux',
'https://raw.githubusercontent.com/TNTwise/REAL-Video-Enhancer/main/bin/glxinfo':'glxinfo',}
                
                
                if self.ui.RifeCheckBox.isChecked() == True and os.path.exists(f'{settings.ModelDir}/rife/') == False:
                        install_modules_dict['https://raw.githubusercontent.com/TNTwise/Rife-Vulkan-Models/main/rife-ncnn-vulkan'] = 'rife-ncnn-vulkan'
                if self.ui.RifeCheckBox.isChecked() == False:
                        
                        os.system(f'rm -rf "{settings.ModelDir}/rife/"')
                if self.ui.RealESRGANCheckBox.isChecked() == True and os.path.exists(f'{settings.ModelDir}/realesrgan') == False:
                        install_modules_dict['https://raw.githubusercontent.com/TNTwise/Rife-Vulkan-Models/main/realesrgan-ncnn-vulkan-20220424-ubuntu.zip'] = 'realesrgan-ncnn-vulkan-20220424-ubuntu.zip'
                if self.ui.RealESRGANCheckBox.isChecked() == False:
                        os.system(f'rm -rf "{settings.ModelDir}/realesrgan/"')
                if self.ui.Waifu2xCheckBox.isChecked() == True and os.path.exists(f'{settings.ModelDir}/waifu2x') == False:
                        install_modules_dict['https://github.com/nihui/waifu2x-ncnn-vulkan/releases/download/20220728/waifu2x-ncnn-vulkan-20220728-ubuntu.zip'] = 'waifu2x-ncnn-vulkan-20220728-ubuntu.zip'
                if self.ui.Waifu2xCheckBox.isChecked() == False:
                        os.system(f'rm -rf "{settings.ModelDir}/waifu2x/"')
                if self.ui.CainCheckBox.isChecked() == True and os.path.exists(f'{settings.ModelDir}/ifrnet') == False:
                          install_modules_dict['https://github.com/nihui/ifrnet-ncnn-vulkan/releases/download/20220720/ifrnet-ncnn-vulkan-20220720-ubuntu.zip'] = 'ifrnet-ncnn-vulkan-20220720-ubuntu.zip'
                if self.ui.CainCheckBox.isChecked() == False:
                         os.system(f'rm -rf "{settings.ModelDir}/ifrnet/"')
                
                if self.ui.RealCUGANCheckBox.isChecked() == True and os.path.exists(f'{settings.ModelDir}/realcugan') == False:
                        install_modules_dict['https://github.com/nihui/realcugan-ncnn-vulkan/releases/download/20220728/realcugan-ncnn-vulkan-20220728-ubuntu.zip'] = 'realcugan-ncnn-vulkan-20220728-ubuntu.zip'
                if self.ui.RealCUGANCheckBox.isChecked() == False:
                        os.system(f'rm -rf "{settings.ModelDir}/realcugan/"')
                for i in rife_install_list:
                        if os.path.exists(f'{settings.ModelDir}/rife/rife-ncnn-vulkan') == False:
                                install_modules_dict['https://raw.githubusercontent.com/TNTwise/Rife-Vulkan-Models/main/rife-ncnn-vulkan'] = 'rife-ncnn-vulkan'
                        if os.path.exists(f'{settings.ModelDir}/rife/{i}') == False:
                                install_modules_dict[f'https://raw.githubusercontent.com/TNTwise/Rife-Vulkan-Models/main/{i}.tar.gz'] = f'{i}.tar.gz'
                if rife_install_list == [] and self.ui.RifeCheckBox.isChecked() and os.path.exists(f'{settings.ModelDir}/rife') == False:
                        install_modules_dict['https://raw.githubusercontent.com/TNTwise/Rife-Vulkan-Models/main/rife-ncnn-vulkan'] = 'rife-ncnn-vulkan'
                        install_modules_dict[f'https://raw.githubusercontent.com/TNTwise/Rife-Vulkan-Models/main/rife-v4.6.tar.gz'] = f'rife-v4.6.tar.gz'
                
                QApplication.closeAllWindows()

                return 0
    class ChooseModels(QtWidgets.QMainWindow):
            def __init__(self,parent):
                super(ChooseModels, self).__init__()
                self.ui = SelectModels.Ui_MainWindow()
                self.ui.setupUi(self)
                self.pinFunctions()
                self.show()
                self.main = parent
            def showDialogBox(self,message,displayInfoIcon=False):
                icon = QIcon(f"{thisdir}/icons/Rife-ESRGAN-Video-Settings - Info.png")
                msg = QMessageBox()
                msg.setWindowTitle(" ")
                if displayInfoIcon == True:
                    msg.setIconPixmap(icon.pixmap(32, 32))
                msg.setText(f"{message}")

                msg.exec_()
            def pinFunctions(self):
                
                self.ui.next.hide()
                rife_pin_functions(self)
                try:
                    with open(f'{thisdir}/models.txt', 'r') as f:
                        for checkbox,option_name in rife_checkboxes(self):
                            
                            if option_name in os.listdir(f'{thisdir}/models/rife/'):
                                checkbox.setChecked(True)
                    
                        for i in f.readlines():
                               print(i)
                               i=i.replace('\n','')
                               for checkbox,option_name in rife_checkboxes(self):
                                    if option_name in i:
                                        checkbox.setChecked(True)
                except:
                     pass
            def checkbox_state_changed(self):
            
                rife_install_list = []

                

                for checkbox, option_name in rife_checkboxes(self):
                    if checkbox.isChecked():
                        rife_install_list.append(option_name)
                
                with open(f'{thisdir}/models.txt', 'w') as f:
                    
                    for option in rife_install_list:
                        f.write(option + '\n')
                



    import src.theme as theme
    app = QtWidgets.QApplication(sys.argv)
    theme.set_theme(app)


    window = ChooseAI()

    app.exec_()


    
    
    class Downloading(QtWidgets.QMainWindow):
                def __init__(self):
                    super(Downloading, self).__init__()
                    self.ui = DownloadUI.Ui_MainWindow()
                    self.ui.setupUi(self)
                    self.show()
                    self.nextfunction()
                    self.setWindowIcon(QIcon(f'{thisdir}/icons/logo v1.png'))

                def showDialogBox(self,message,displayInfoIcon=False):
                    icon = QIcon(f"{thisdir}/icons/Rife-ESRGAN-Video-Settings - Info.png")
                    msg = QMessageBox()
                    msg.setWindowTitle(" ")
                    if displayInfoIcon == True:
                        msg.setIconPixmap(icon.pixmap(32, 32))
                    msg.setText(f"{message}")

                    msg.exec_()

                def nextfunction(self):

                    logo = QIcon(f"{thisdir}/icons/logo v1.png")

                    self.ui.logoPreview.setPixmap(logo.pixmap(256,256))

                    self.obj = Worker(self)
                    self.thread = QThread()
                    self.obj.intReady.connect(self.on_count_changed)
                    self.obj.moveToThread(self.thread)
                    self.obj.finished.connect(self.thread.quit)
                    self.thread.started.connect(self.obj.install_modules)
                    self.thread.start()
                    self.obj.finished.connect(self.start_main)
                def on_count_changed(self,list):
                    downloaded_data = list[0]
                    total_data = list[1]
                    self.ui.downloadProgressBar.setMaximum(total_data)
                    self.ui.downloadProgressBar.setValue(downloaded_data)
                    downloaded_data_gb = str(downloaded_data/1000000000)[:4]
                    total_data_gb = str(total_data/1000000000)[:4]

                    self.ui.gbLabel.setText(f'{downloaded_data_gb}/{total_data_gb}GB')
                def start_main(self):
                    
                         
                    for i in os.listdir(f'{thisdir}/files/'):
                        if os.path.exists(f'{thisdir}/bin/') == False:
                            os.mkdir(f'{thisdir}/bin/')
                        if i == 'ffmpeg':
                            os.system(f'chmod +x "{thisdir}/files/ffmpeg"')
                            os.system(f'mv "{thisdir}/files/ffmpeg" "{thisdir}/bin/"')
                        if i == 'yt-dlp_linux':
                            os.system(f'chmod +x "{thisdir}/files/yt-dlp_linux"')
                            os.system(f'mv "{thisdir}/files/yt-dlp_linux" "{thisdir}/bin/"')
                        if i == 'glxinfo':
                            os.system(f'chmod +x "{thisdir}/files/glxinfo"')
                            os.system(f'mv "{thisdir}/files/glxinfo" "{thisdir}/bin/"')
                        
                        
                        print(i)
                        if '.zip' in i:

                            with ZipFile(f'{thisdir}/files/{i}', 'r') as zip_ref:
                                name=i.replace('.zip','')
                                original_ai_name_ncnn_vulkan = re.findall(r'[\w]*-ncnn-vulkan', name)[0]
                                original_ai_name = original_ai_name_ncnn_vulkan.replace('-ncnn-vulkan','')
                                print(original_ai_name)


                                zip_ref.extractall(f'{thisdir}/files/')

                            os.system(f'mv "{thisdir}/files/{name}" "{settings.ModelDir}/{original_ai_name}"')
                            os.system(f'chmod +x "{settings.ModelDir}/{original_ai_name}/{original_ai_name_ncnn_vulkan}"')

                        if '.tar.gz' in i:
                            with tarfile.open(f'{thisdir}/files/{i}','r') as f:
                                f.extractall(f'{settings.ModelDir}/rife/')


                    
                    clear_files()
                    if check_for_individual_models != None:
                        if check_if_online():
                            QApplication.closeAllWindows()

                            return 0
                        else:
                            exit()
                    else:
                        failed_download(self)
                        
                        exit()
                    
    import src.theme as theme

    


    window = Downloading()
    app.exec_()
    app =None 
    if os.path.isfile(f'{settings.ModelDir}/rife/rife-ncnn-vulkan') == True:
        QApplication.closeAllWindows()
    else:
        for file in os.listdir(f'{thisdir}/files'):
            if '.txt' not in file:
                os.system(f'rm -rf "{thisdir}/files/{file}"')
        exit() # this happens if program abruptly stops while downloading
'''else:
    exit()'''
