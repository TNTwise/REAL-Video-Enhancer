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
                rife_install_list= []
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
                    with open(f'{thisdir}/src/getModels/models.txt', 'w') as f:
                        pass
                    with open(f'{thisdir}/src/getModels/models.txt', 'a') as f:
                        for i in rife_install_list:
                            f.write(i + '\n')
                    os.system(f'python3 {thisdir}/src/getModels/get_ai_models.py')
                    return 0
                    
                
                
                

    

    app = QtWidgets.QApplication(sys.argv)
    
    
        
    window = ChooseModels()
    app.exec_()
    if os.path.isfile(f'{thisdir}/realesrgan-vulkan-models/realesrgan-ncnn-vulkan') == False or os.path.isfile(f'{thisdir}/rife-vulkan-models/rife-ncnn-vulkan') == False:
        exit()