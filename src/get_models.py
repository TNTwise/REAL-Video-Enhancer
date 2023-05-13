import os
thisdir = os.getcwd()
import sys
import requests
import re
from zipfile import ZipFile
from PyQt5 import QtWidgets, uic
from time import sleep
from PyQt5.QtWidgets import QApplication, QPushButton, QWidget, QHBoxLayout, QProgressBar, QVBoxLayout, QMessageBox
from src.settings import *
from src.return_data import *
from threading import Thread

if os.path.exists(f'{thisdir}/Real-ESRGAN/') == False or os.path.exists(f'{thisdir}/rife-vulkan-models/') == False:
    
    class PopUpProgressB(QtWidgets.QMainWindow):
        def __init__(self,model):

            try:
                requests.get('https://www.github.com')

                super(PopUpProgressB, self).__init__()
                self.startProgressBar(f'{model}')
                
            except:
                msg = QMessageBox()
                msg.setWindowTitle(" ")
                msg.setText(f"You are offline, please connect to the internet to download the models or download the offline binary.")
                sys.exit(msg.exec_())

        def startProgressBar(self,model):

            self.pbar = QProgressBar(self)
            self.pbar.setGeometry(30, 40, 500, 75)
            self.layout = QVBoxLayout()
            self.layout.addWidget(self.pbar)
            self.setLayout(self.layout)
            self.setGeometry(300, 300, 550, 100)
            self.setWindowTitle(f'Downloading {model} Models')
            self.show()

            Thread(target=lambda: self.show_loading_window(f'{model}')).start()     
        
        
        def latest_rife(self):
            latest = requests.get('https://github.com/nihui/rife-ncnn-vulkan/releases/latest/')
            latest = latest.url
            latest = re.findall(r'[\d]*$', latest)
            latest = latest[0]
            return(latest)
        
        def show_loading_window(self,model):
            
            os.chdir(f"{thisdir}/files/")
            
        
            
            if model == 'Real-ESRGAN':
                file=f"realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
                response = requests.get(f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip", stream=True)
            else:
                file=f"rife-ncnn-vulkan-{self.latest_rife()}-ubuntu.zip"
                response = requests.get(f"https://github.com/nihui/rife-ncnn-vulkan/releases/download/{self.latest_rife()}/rife-ncnn-vulkan-{self.latest_rife()}-ubuntu.zip", stream=True)

            total_size_in_bytes= int(response.headers.get('content-length', 0))
            block_size = 1024 #1 Kibibyte
            
            self.pbar.setMaximum(total_size_in_bytes)
            
            total_block_size = 0
            with open(file, 'wb') as f:
                for data in response.iter_content(block_size):
                    total_block_size += block_size
                    self.pbar.setValue(total_block_size)
                    
                    f.write(data)
                if model == 'Real-ESRGAN':
                    Thread(target=self.get_realesrgan).start()
                else:
                    Thread(target=self.get_rife).start()
                
                    
            
                

        def get_realesrgan(self):

            os.system(f'mkdir -p "{thisdir}/files/realesrgan-ncnn-vulkan-20220424-ubuntu/"')
            os.chdir(f'{thisdir}/files/realesrgan-ncnn-vulkan-20220424-ubuntu/')
            with ZipFile(f'{thisdir}/files/realesrgan-ncnn-vulkan-20220424-ubuntu.zip','r') as f:
                f.extractall()
            os.system(f'mkdir -p "{thisdir}/Real-ESRGAN/models/"')
            os.system(f'mv "{thisdir}/files/realesrgan-ncnn-vulkan-20220424-ubuntu/"* "{thisdir}/Real-ESRGAN/"')
            
            
            
            os.system(f'chmod +x "{thisdir}/Real-ESRGAN/realesrgan-ncnn-vulkan" && rm -rf "{thisdir}/files/realesrgan-ncnn-vulkan-20220424-ubuntu" && rm -rf "{thisdir}/files/realesrgan-ncnn-vulkan-20220424-ubuntu.zip" && rm -rf "{thisdir}/Real-ESRGAN/input.jpg" && rm -rf  "{thisdir}/Real-ESRGAN/input2.jpg" && rm -rf "{thisdir}/Real-ESRGAN/onepiece_demo.mp4"')
            os.chdir(f'{thisdir}')
            self.close()
            if os.path.isfile(f'{thisdir}/files/rife-ncnn-vulkan-{self.latest_rife()}-ubuntu.zip') == False:
                sleep(1)
                import main as main # it will prob call this, but its ok as i am a shit programmer, for some reason, it wont work without the if statement but this seems sketcy at best for a solution
            

        def get_rife(self):

                version = self.latest_rife() # calls latest function which gets the latest version release of rife and returns the latest and the current, if the version file doesnt exist, it updates and creates the file
                latest_ver = version


                os.chdir(f"{thisdir}")
                with ZipFile(f'{thisdir}/files/rife-ncnn-vulkan-{latest_ver}-ubuntu.zip', 'r') as zip_ref:
                    zip_ref.extractall(f'{thisdir}/files/')
                os.system(f'mkdir -p "{thisdir}/rife-vulkan-models"')
                os.system(f'mv "{thisdir}/rife-ncnn-vulkan-{latest_ver}-ubuntu" "{thisdir}/files/"')
                os.system(f'mv "{thisdir}/files/rife-ncnn-vulkan-{latest_ver}-ubuntu/"* "{thisdir}/rife-vulkan-models/"')
                Settings.change_setting(0,'rifeversion', f'{latest_ver}')
                os.system(f'rm -rf "{thisdir}/files/rife-ncnn-vulkan-{latest_ver}-ubuntu.zip"')
                os.system(f'rm -rf "{thisdir}/files/rife-ncnn-vulkan-{latest_ver}-ubuntu"')
                os.system(f'chmod +x "{thisdir}/rife-vulkan-models/rife-ncnn-vulkan"')

                
                self.close()
                sleep(1) #need sleep otherwise core dump :\ this is probably not a good idea....
                    
                import main as main
    
    class StartRife:
        if os.path.isfile(f'{thisdir}/src/rife_models.txt') == True:
            app1 = QApplication(sys.argv)
            if os.path.exists(f'{thisdir}/Real-ESRGAN/') == False:
                main_window = PopUpProgressB('Real-ESRGAN')
            if os.path.exists(f'{thisdir}/rife-vulkan-models/') == False:
                main_window = PopUpProgressB('Rife')
        
            sys.exit(app1.exec_())

        
        
        

