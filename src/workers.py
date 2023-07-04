
from PyQt5 import QtWidgets, uic
import sys
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QMessageBox, QListWidget, QListWidgetItem
from PyQt5.QtGui import QTextCursor
import mainwindow
import os
from threading import *
from src.settings import *
from src.return_data import *
from time import sleep

class pb2X(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    def __init__(self,myvar,render,parent=None):
        
        QThread.__init__(self, parent)
        self.input_file = myvar
        self.videoName = VideoName.return_video_name(f'{self.input_file}')
        self.settings = Settings()
        self.render = render
    def run(self):
        """Long-running task."""
        print('\n\n\n\n')
        
        while ManageFiles.isfolder(f'{self.settings.RenderDir}/{self.videoName}_temp/output_frames/') == False:
            sleep(.1) # has to refresh quickly or small files that interpolate fast do not work
         

        total_input_files = len(os.listdir(f'{self.settings.RenderDir}/{self.videoName}_temp/input_frames/'))
        total_output_files = total_input_files * 2
        # fc is the total file count after interpolation
        #Could use this instead of just os.listdir
        '''while os.path.exists(f'{self.render_folder}/{self.videoName}_temp/output_frames/'):


            try:

                   #Have to make more optimized sorting alg here 

                    if last_file != None:
                        iteration=int(str(last_file).replace('.png',''))
                        while os.path.exists(f'{self.render_folder}/{self.videoName}_temp/output_frames/{str(iteration).zfill(8)}.png') == True:
                            iteration+=1

                        last_file=f'{str(iteration).zfill(8)}.png'
                        print(last_file)
                    else:
                        files = os.listdir(f'{self.render_folder}/{self.videoName}_temp/output_frames/')
                        files.sort()
                        last_file = files[-1]

                    self.imageDisplay = f"{self.render_folder}/{self.videoName}_temp/output_frames/{last_file}"
            except:
                    self.imageDisplay = None
                    self.ui.imagePreview.clear()
                    self.ui.imagePreviewESRGAN.clear()
            sleep(.5)
'''
        
        
        while ManageFiles.isfolder(f'{self.settings.RenderDir}/{self.videoName}_temp/') == True:
                if ManageFiles.isfolder(f'{self.settings.RenderDir}/{self.videoName}_temp/output_frames/') == True:
                
                    files_processed = len(os.listdir(f'{self.settings.RenderDir}/{self.videoName}_temp/output_frames/'))
                    
                    sleep(1)
                    
                        
                    
                    self.progress.emit(files_processed)
        sleep(1)
        self.finished.emit()


class showLogs(QObject):
    finished = pyqtSignal()
    extractionProgress = pyqtSignal(int)
    
    def __init__(self,parent, videoName):
        QThread.__init__(self, parent)
        self.videoName = videoName
        self.settings = Settings()
    def run(self):
        """Long-running task."""
        
        while os.path.exists(f'{self.settings.RenderDir}/{self.videoName}_temp/output_frames/') == False:
            if os.path.exists(f'{self.settings.RenderDir}/{self.videoName}_temp/input_frames/'):
                files_extracted = len(os.listdir(f'{self.settings.RenderDir}/{self.videoName}_temp/input_frames/'))
                self.extractionProgress.emit(files_extracted)
        self.finished.emit()
