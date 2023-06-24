import os
import requests
# Print iterations progress
thisdir = os.getcwd()

    
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration >= total: 
        print('\nDownloaded!')
import sys
import time
from PyQt5.QtCore import QThread, pyqtSignal, QObject, pyqtSlot
from PyQt5.QtWidgets import QApplication, QPushButton, QWidget, QHBoxLayout, QProgressBar, QVBoxLayout
from zipfile import *

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Widget")
        self.h_box = QHBoxLayout(self)
        self.main_window_button = QPushButton("Start")
        self.popup = PopUpProgressB()
        self.main_window_button.clicked.connect(self.popup.show)
        
        self.h_box.addWidget(self.main_window_button)
        self.setLayout(self.h_box)
        


class Worker(QObject):
    finished = pyqtSignal()
    intReady = pyqtSignal(int)

    @pyqtSlot()
    def proc_counter(self):  # A slot takes no params
        try:
            os.mkdir(f'{thisdir}/files/')
        except:
            os.chdir(f"{thisdir}/files/")
        
        file=f"rife-ncnn-vulkan-20221029-ubuntu"   
        model='rife'
        
        for model_range in range(2):
        
            if file == f"rife-ncnn-vulkan-20221029-ubuntu":
                response = requests.get(f"https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-ubuntu.zip", stream=True)
            else:
                response = requests.get(f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip", stream=True)
            total_size_in_bytes= int(response.headers.get('content-length', 0))
            block_size = 1024 #1 Kibibyte
            
            
            
            total_block_size = 0
            with open(f'{thisdir}/files/{file}.zip', 'wb') as f:
                for data in response.iter_content(block_size):
                    total_block_size += block_size
                    
                    printProgressBar(total_block_size/total_size_in_bytes*100,100)
                    f.write(data)
                    self.intReady.emit(int(total_block_size/total_size_in_bytes*100))
                    if int(total_block_size/total_size_in_bytes*100) == 100:
                        os.chdir(f'{thisdir}')
                        
                        break        
            with ZipFile(f'{thisdir}/files/{file}.zip', 'r') as zip_ref:
                os.mkdir(f'{thisdir}/files/{file}')
                if model_range == 0:
                    zip_ref.extractall(f'{thisdir}/files/')
                else:
                    zip_ref.extractall(f'{thisdir}/files/{file}')
            os.system(f'mkdir -p "{thisdir}/{model}-vulkan-models"')
            
            os.system(f'chmod +x "{thisdir}/files/{file}/{model}-ncnn-vulkan"')
            os.system(f'mv "{thisdir}/files/{file}/"* "{thisdir}/{model}-vulkan-models/"')
            

            os.system(f'rm -rf "{thisdir}/files/{file}.zip"')
            os.system(f'rm -rf "{thisdir}/files/{file}"')
            
            rife_model_list=[]
            save_list=[]
            with open(f'{thisdir}/src/getModels/models.txt', 'r') as f:
                for line in f:
                    line = line.replace('\n','')
                    rife_model_list.append(line)
            
            for i in os.listdir(f'{thisdir}/rife-vulkan-models/'):
                for j in rife_model_list:
                    if i == j:
                        save_list.append(i)
            save_list.append('rife-ncnn-vulkan')
            
            for i in os.listdir(f'{thisdir}/rife-vulkan-models/'):
                
                if i not in save_list:
                    
                    os.system(f'rm -rf {thisdir}/rife-vulkan-models/{i}')
            os.system(f'rm -rf {thisdir}/src/getModels/models.txt')
            
            file=f"realesrgan-ncnn-vulkan-20220424-ubuntu" 
            model='realesrgan'
        self.finished.emit()
            


class PopUpProgressB(QWidget):

    def __init__(self):
        super().__init__()
        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(30, 40, 100, 75)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.pbar)
        self.setLayout(self.layout)
        self.setGeometry(300, 300, 550, 100)
        self.setWindowTitle('Downloading Models')
        self.show()

        self.obj = Worker()
        self.thread = QThread()
        self.obj.intReady.connect(self.on_count_changed)
        self.obj.moveToThread(self.thread)
        self.obj.finished.connect(self.thread.quit)
        self.thread.started.connect(self.obj.proc_counter)
        self.thread.start()
        self.obj.finished.connect(exit)
        
    def on_count_changed(self, value):
        self.pbar.setValue(value)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    sys.exit(app.exec_())