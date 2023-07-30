
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


class downloadVideo(QObject):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(str)
    addRes = pyqtSignal(str)
    def __init__(self,originalSelf,url,parent=None):
        self.originalSelf = originalSelf
        self.url = url
        QThread.__init__(self, parent)
    def run(self):
        try:
                print(self.originalSelf.ui.plainTextEdit.toPlainText())
                result = subprocess.run([f'{thisdir}/bin/yt-dlp_linux', '-F', self.url], capture_output=True, text=True)
                
                if result.returncode == 0:
                    stdout_lines = result.stdout.splitlines()
                    resolutions_list = []
                    self.dict_res_id_fps = {}
                    fps_list=[]
                    for line in stdout_lines:
                         if 'FPS' in line:
                            fps_index = line.find('FPS')
                            print(fps_index)
                            break
                    for line in reversed(stdout_lines):
                        
                        if 'mp4' in line:
                            
                            resolution = re.findall(r'[\d]*x[\d]*',line)
                            if len(resolution) > 0:
                                if resolution[0] not in resolutions_list:
                                    res=resolution[0]
                                    resolutions_list.append(res)
                                    id=line[:3]
                                    fps=(line[fps_index:fps_index+3])
                                    self.dict_res_id_fps[res] = [id,fps]
                                    self.addRes.emit(res)
                    self.originalSelf.duration = self.originalSelf.get_youtube_video_duration(self.url)
                    name = self.originalSelf.get_youtube_video_name(self.url)
                    self.originalSelf.main.input_file = f'{thisdir}/{name}.mp4'
                    self.originalSelf.main.videoName = f'{name}.mp4'
                    self.finished.emit(self.dict_res_id_fps)
                else:
                    self.progress.emit(result.stderr)
        except Exception as e:
                print(e)
                