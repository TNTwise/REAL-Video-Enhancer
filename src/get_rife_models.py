import os
thisdir = os.getcwd()
import sys
import requests
import re
from zipfile import ZipFile
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QPushButton, QWidget, QHBoxLayout, QProgressBar, QVBoxLayout, QMessageBox
from src.settings import *
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        msg = QMessageBox()
        msg.setWindowTitle(" ")
        msg.setText(f"You are offline, please connect to the internet to download the models or download the offline binary.")
        msg.exec_()
        exit()

app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
class PopUpProgressB(QWidget):

    def progressBar(self,model):
        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(30, 40, 500, 75)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.pbar)
        self.setLayout(self.layout)
        self.setGeometry(300, 300, 550, 100)
        self.setWindowTitle(f'{model}')
        self.show()

       

    def on_count_changed(self, value):
        self.pbar.setValue(value)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = PopUpProgressB()
    
    sys.exit(app.exec_())

class get_all_models:
    def __init__(self):
         self.get_rife()
         self.get_realesrgan()
         os.chdir(f"{thisdir}")
         os.execv(sys.executable, ['python'] + sys.argv)

    def latest_rife(self):
        latest = requests.get('https://github.com/nihui/rife-ncnn-vulkan/releases/latest/') 
        latest = latest.url
        latest = re.findall(r'[\d]*$', latest)
        latest = latest[0]
        print(latest)
        return(latest)
    
    def show_loading_window(self, model):
        
        
        
        os.chdir(f"{thisdir}/files/")
        
        
        

        version = self.latest_rife() # calls latest function which gets the latest version release of rife and returns the latest and the current, if the version file doesnt exist, it updates and creates the file
        latest_ver = version
        try:
            if model == 'rife':
                #message = Label(self.loading_window, text='Downloading Rife Models',font=('Ariel', '12'),bg=bg,fg=fg)
                file=f"rife-ncnn-vulkan-{latest_ver}-ubuntu.zip"
                response = requests.get(f"https://github.com/nihui/rife-ncnn-vulkan/releases/download/{latest_ver}/rife-ncnn-vulkan-{latest_ver}-ubuntu.zip", stream=True)
            if model == 'realesrgan':
                file=f"realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
                
                response = requests.get(f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip", stream=True)
                #message = Label(self.loading_window, text='Downloading Real-ESRGAN Models',font=('Ariel', '12'),bg=bg,fg=fg)
            total_size_in_bytes= int(response.headers.get('content-length', 0))
            block_size = 1024 #1 Kibibyte
            #progressbar = ttk.Progressbar(self.loading_window,orient='horizontal', length=400, mode="determinate",maximum=total_size_in_bytes,value=0)
            #message.grid(column=0,row=0)
            #progressbar.grid(column=0, row=1)
            # Add progressbar updater
            #progressbar["maximum"]=total_size_in_bytes
            
            with open(file, 'wb') as f:
                for data in response.iter_content(block_size):
                        
                    #progressbar['value'] += block_size
                    #progressbar.update()
                    f.write(data)
                
                return
            
            
                
            
            
        
        except:
            
            app.exec_()
            exit()
            #message = Label(self.loading_window, text='Your are offline, Please Reconnect to the internet\n or download the offline binary.',font=('Ariel', '12'),bg=bg,fg=fg)
            
             # this is offline mode
            #message.grid(column=0,row=0)
            

        #self.loading_window.mainloop()


    def get_rife(self): 
        
        if os.path.exists(f"{thisdir}/rife-vulkan-models/") == False:
            
            ManageFiles.create_folder(f"{thisdir}/files")
            self.show_loading_window('rife')
            version = self.latest_rife() # calls latest function which gets the latest version release of rife and returns the latest and the current, if the version file doesnt exist, it updates and creates the file
            latest_ver = version
            os.chdir(f"{thisdir}/files/")
            with ZipFile(f'rife-ncnn-vulkan-{latest_ver}-ubuntu.zip','r') as f:
                f.extractall()
            os.chdir(f"{thisdir}")
            os.system(f'rm -rf "{thisdir}/rife-vulkan-models"')
            os.system(f'mkdir -p "{thisdir}/rife-vulkan-models"')
            os.system(f'mv "{thisdir}/rife-ncnn-vulkan-{latest_ver}-ubuntu" "{thisdir}/files/"')
            os.system(f'mv "{thisdir}/files/rife-ncnn-vulkan-{latest_ver}-ubuntu/"* "{thisdir}/rife-vulkan-models/"')
            Settings.change_setting(0,'rifeversion', f'{latest_ver}')
            os.system(f'rm -rf "{thisdir}/files/rife-ncnn-vulkan-{latest_ver}-ubuntu.zip"')
            os.system(f'rm -rf "{thisdir}/files/rife-ncnn-vulkan-{latest_ver}-ubuntu"')
            os.system(f'chmod +x "{thisdir}/rife-vulkan-models/rife-ncnn-vulkan"')
            
            
    def get_realesrgan(self):
        
        if os.path.exists(f"{thisdir}/Real-ESRGAN/") == False:
            
        
            
            
            
            self.show_loading_window('realesrgan')
            os.system(f'mkdir -p "{thisdir}/files/realesrgan-ncnn-vulkan-20220424-ubuntu/"')
            os.chdir(f'{thisdir}/files/realesrgan-ncnn-vulkan-20220424-ubuntu/')
            with ZipFile(f'{thisdir}/files/realesrgan-ncnn-vulkan-20220424-ubuntu.zip','r') as f:
                f.extractall()
            os.system(f'mkdir -p "{thisdir}/Real-ESRGAN/models/"')
            os.system(f'mv "{thisdir}/files/realesrgan-ncnn-vulkan-20220424-ubuntu/"* "{thisdir}/Real-ESRGAN/"')
            
            
            
            os.system(f'chmod +x "{thisdir}/Real-ESRGAN/realesrgan-ncnn-vulkan" && rm -rf "{thisdir}/files/realesrgan-ncnn-vulkan-20220424-ubuntu" && rm -rf "{thisdir}/files/realesrgan-ncnn-vulkan-20220424-ubuntu.zip" && rm -rf "{thisdir}/Real-ESRGAN/input.jpg" && rm -rf  "{thisdir}/Real-ESRGAN/input2.jpg" && rm -rf "{thisdir}/Real-ESRGAN/onepiece_demo.mp4"')
           
        os.system(f'rm -rf "{thisdir}/temp/"')
