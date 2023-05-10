import os
thisdir = os.getcwd()
import sys
import requests
import re
from zipfile import ZipFile
from PyQt5 import QtWidgets, uic

from PyQt5.QtWidgets import QApplication, QPushButton, QWidget, QHBoxLayout, QProgressBar, QVBoxLayout, QMessageBox
from src.settings import *
from src.return_data import *
from threading import Thread
import src.get_rife_models as get_rife_models

if os.path.exists(f'{thisdir}/Real-ESRGAN/') == False:
    class PopUpProgressB(QtWidgets.QMainWindow):
        def __init__(self):

            try:
                requests.get('https://www.github.com')

                super(PopUpProgressB, self).__init__()

                self.progressBarRealSR()
            except:
                msg = QMessageBox()
                msg.setWindowTitle(" ")
                msg.setText(f"You are offline, please connect to the internet to download the models or download the offline binary.")
                sys.exit(msg.exec_())
                
        def progressBarRealSR(self):
            
            self.pbar = QProgressBar(self)
            self.pbar.setGeometry(30, 40, 500, 75)
            self.layout = QVBoxLayout()
            self.layout.addWidget(self.pbar)
            self.setLayout(self.layout)
            self.setGeometry(300, 300, 550, 100)
            self.setWindowTitle(f'Downloading Real-ESRGAN Models')
            self.show()
            
            Thread(target=self.show_loading_window).start()
        
        
        
        
        def show_loading_window(self):
            
            os.chdir(f"{thisdir}/files/")
            
        
            try:
                
                file=f"realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
                response = requests.get(f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip", stream=True)
                
                total_size_in_bytes= int(response.headers.get('content-length', 0))
                block_size = 1024 #1 Kibibyte
                
                self.pbar.setMaximum(total_size_in_bytes)
                
                total_block_size = 0
                with open(file, 'wb') as f:
                    for data in response.iter_content(block_size):
                        total_block_size += block_size
                        self.pbar.setValue(total_block_size)
                        
                        f.write(data)

                    Thread(target=self.get_realesrgan).start()
                    
                    self.close()
                    
            except:
                msg = QMessageBox()
                msg.setWindowTitle(" ")
                msg.setText(f"You are offline, please download the offline binary or reconnect to the internet.")
                msg.exec_()
                

        def get_realesrgan(self):

            os.system(f'mkdir -p "{thisdir}/files/realesrgan-ncnn-vulkan-20220424-ubuntu/"')
            os.chdir(f'{thisdir}/files/realesrgan-ncnn-vulkan-20220424-ubuntu/')
            with ZipFile(f'{thisdir}/files/realesrgan-ncnn-vulkan-20220424-ubuntu.zip','r') as f:
                f.extractall()
            os.system(f'mkdir -p "{thisdir}/Real-ESRGAN/models/"')
            os.system(f'mv "{thisdir}/files/realesrgan-ncnn-vulkan-20220424-ubuntu/"* "{thisdir}/Real-ESRGAN/"')
            
            
            
            os.system(f'chmod +x "{thisdir}/Real-ESRGAN/realesrgan-ncnn-vulkan" && rm -rf "{thisdir}/files/realesrgan-ncnn-vulkan-20220424-ubuntu" && rm -rf "{thisdir}/files/realesrgan-ncnn-vulkan-20220424-ubuntu.zip" && rm -rf "{thisdir}/Real-ESRGAN/input.jpg" && rm -rf  "{thisdir}/Real-ESRGAN/input2.jpg" && rm -rf "{thisdir}/Real-ESRGAN/onepiece_demo.mp4"')
            os.chdir(f'{thisdir}')
            if ManageFiles.isfolder(f'{thisdir}/rife-vulkan-models/') == False:
                import src.get_rife_models as get_rife_models
            import main as main
    
    class StartRealSR:
        
            app = QApplication(sys.argv)
            main_window = PopUpProgressB()
                
            sys.exit(app.exec_())

