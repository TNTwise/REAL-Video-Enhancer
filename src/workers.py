
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
    def __init__(self,myvar,parent=None):
        
        QThread.__init__(self, parent)
        self.input_file = myvar
        self.videoName = VideoName.return_video_name(f'{self.input_file}')
        self.settings = Settings()
    
    def run(self):
        """Long-running task."""
        print('\n\n\n\n')
        
        while ManageFiles.isfolder(f'{self.settings.RenderDir}/{self.videoName}_temp/output_frames/') == False:
            sleep(1)
        

        total_input_files = len(os.listdir(f'{self.settings.RenderDir}/{self.videoName}_temp/input_frames/'))
        total_output_files = total_input_files * 2
        
        sleep_time=int(.05*(VideoName.return_video_frame_count(self.input_file)/VideoName.return_video_framerate(self.input_file)))
        print(sleep_time)
        
        
        while ManageFiles.isfolder(f'{self.settings.RenderDir}/{self.videoName}_temp/') == True:
                if ManageFiles.isfolder(f'{self.settings.RenderDir}/{self.videoName}_temp/output_frames/') == True:
                
                    files_processed = len(os.listdir(f'{self.settings.RenderDir}/{self.videoName}_temp/output_frames/'))
                    try:
                        latest_image = self.get_latest_image()
                    except:
                        latest_image= None
                    
                    sleep(.1)
                    
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
