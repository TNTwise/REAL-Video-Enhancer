# WORK IN PROGRESS!!!!

import os
thisdir = os.getcwd()
import sys
import requests
import re
from zipfile import ZipFile
from PyQt5 import QtWidgets, uic
from time import sleep
from PyQt5.QtWidgets import QApplication, QPushButton, QWidget, QHBoxLayout, QProgressBar, QVBoxLayout, QMessageBox
from PyQt5.QtGui import QIcon
import src.messages
from threading import Thread
import src.getModels.SelectModels as SelectModels
import src.getModels.Download as DownloadUI
global rife_install_list
from PyQt5.QtCore import QThread, pyqtSignal, QObject, pyqtSlot

rife_install_list=[]
class Worker(QObject):
    finished = pyqtSignal()
    intReady = pyqtSignal(list)
    
                    
    @pyqtSlot()
    def install_modules(self):
                    install_modules_dict={
                                        
'https://github.com/nihui/realcugan-ncnn-vulkan/releases/download/20220728/realcugan-ncnn-vulkan-20220728-ubuntu.zip':'realcugan-ncnn-vulkan-20220728-ubuntu.zip',
'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip':'realesrgan-ncnn-vulkan-20220424-ubuntu.zip',
'https://github.com/nihui/cain-ncnn-vulkan/releases/download/20220728/cain-ncnn-vulkan-20220728-ubuntu.zip':'cain-ncnn-vulkan-20220728-ubuntu.zip',
'https://raw.githubusercontent.com/TNTwise/Rife-Vulkan-Models/main/rife-ncnn-vulkan':'rife-ncnn-vulkan'}
                    for i in rife_install_list:
                                        install_modules_dict[f'https://raw.githubusercontent.com/TNTwise/Rife-Vulkan-Models/main/{i}.tar.gz'] = f'{i}.tar.gz'
                    total_size_in_bytes=0
                    data_downloaded=0
                    for link,name in install_modules_dict.items():
                        response = requests.get(link, stream=True)
                        total_size_in_bytes+= int(response.headers.get('content-length', 0))
                    
                    for link,name in install_modules_dict.items():
                        response = requests.get(link, stream=True)
                        with open(f'{thisdir}/files/{name}', 'wb') as f:
                                for data in response.iter_content(1024):
                                    f.write(data)
                                    data_downloaded+=1024
                                    self.intReady.emit([int(data_downloaded),total_size_in_bytes]) # sends back data to main thread
                    
                    
                    
                    self.finished.emit()

if os.path.isfile(f'{thisdir}/realesrgan-vulkan-models/realesrgan-ncnn-vulkan') == False or os.path.isfile(f'{thisdir}/rife-vulkan-models/rife-ncnn-vulkan') == False:
    
    class ChooseModels(QtWidgets.QMainWindow):
            def __init__(self):
                super(ChooseModels, self).__init__()
                self.ui = SelectModels.Ui_MainWindow()
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
                
                self.ui.next.clicked.connect(self.nextfunction)
            def nextfunction(self):
                
                if self.ui.rife.isChecked() == True:
                    rife_install_list.append('rife')
                if self.ui.rifeanime.isChecked() == True:
                    rife_install_list.append('rife-anime')
                if self.ui.rifehd.isChecked() == True:
                    rife_install_list.append('rife-HD')
                if self.ui.rifeuhd.isChecked() == True:
                    rife_install_list.append('rife-UHD')
                if self.ui.rife2.isChecked() == True:
                    rife_install_list.append('rife-v2')
                if self.ui.rife23.isChecked() == True:
                    rife_install_list.append('rife-v2.3')
                if self.ui.rife24.isChecked() == True:
                    rife_install_list.append('rife-v2.4')
                if self.ui.rife30.isChecked() == True:
                    rife_install_list.append('rife-v3.0')
                if self.ui.rife31.isChecked() == True:
                    rife_install_list.append('rife-v3.1')
                if self.ui.rife4.isChecked() == True:
                    rife_install_list.append('rife-v4')
                if self.ui.rife46.isChecked() == True:
                    rife_install_list.append('rife-v4.6')
                if len(rife_install_list) == 0:
                    src.messages.no_downloaded_models(self)
                else:
                    QApplication.closeAllWindows()
                    
                    
                    return 0
                    
                
                
                

    import src.theme as theme
    
    app = QtWidgets.QApplication(sys.argv)
    theme.set_theme(app)
    
    
    window = ChooseModels()
    app.exec_()
    if len(rife_install_list) > 0:
        class Downloading(QtWidgets.QMainWindow):
                def __init__(self):
                    super(Downloading, self).__init__()
                    self.ui = DownloadUI.Ui_MainWindow()
                    self.ui.setupUi(self)
                    self.show()
                    self.nextfunction()
                    
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
                    
                    self.obj = Worker()
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
                    QApplication.closeAllWindows()
                    return 0

        import src.theme as theme
        
        app1 = QtWidgets.QApplication(sys.argv)
        theme.set_theme(app1)
        
        
        window = Downloading()
        app1.exec_()
        
        
    else:
        exit()
