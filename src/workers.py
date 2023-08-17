
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
import src.thisdir
thisdir = src.thisdir.thisdir()
class pb2X(QObject):
    finished = pyqtSignal()
    image_progress = pyqtSignal(str)
    progress = pyqtSignal(int)
    def __init__(self,myvar,render,main,parent=None):
        
        QThread.__init__(self, parent)
        self.input_file = myvar
        self.videoName = VideoName.return_video_name(f'{self.input_file}')
        self.settings = Settings()
        self.render = render
        self.main = main
    def run(self):
        """Long-running task."""
        print('\n\n\n\n')
        
        while ManageFiles.isfolder(f'{self.settings.RenderDir}/{self.videoName}_temp/output_frames/') == False:
            sleep(.1) # has to refresh quickly or small files that interpolate fast do not work
         

        total_input_files = len(os.listdir(f'{self.settings.RenderDir}/{self.videoName}_temp/input_frames/'))
        total_output_files = total_input_files * 2
        # fc is the total file count after interpolation
        #Could use this instead of just os.listdir
        
        
        
        while ManageFiles.isfolder(f'{self.settings.RenderDir}/{self.videoName}_temp/') == True:
                if ManageFiles.isfolder(f'{self.settings.RenderDir}/{self.videoName}_temp/output_frames/') == True:
                
                    files_processed = len(os.listdir(f'{self.settings.RenderDir}/{self.videoName}_temp/output_frames/'))
                    
                    
                    sleep(.1)
                    
                    self.progress.emit(files_processed)
                    self.main.imageDisplay=f'{self.settings.RenderDir}/{self.main.videoName}_temp/output_frames/{str(files_processed-int(self.settings.VRAM)-1).zfill(8)}{self.settings.Image_Type}' # sets behind to stop corrupted jpg error
                    
                    if self.main.imageDisplay != None:

                        try:
                            if os.path.exists(self.main.imageDisplay):
                                self.image_progress.emit('1')
                                
                                width = self.main.width()
                                height = self.main.height()
                                
                                self.main.width1=int(width/1.4)
                                self.main.height1=int(self.main.width1/self.main.aspectratio)
                                if self.main.height1 >= height/1.4:
                                    
                                    self.main.height1=int(height/1.4)
                                    self.main.width1=int(self.main.height1/(self.main.videoheight/self.main.videowidth))
                                try:
                                    if os.path.exists(self.main.imageDisplay):
                                        
                                        
                                        
                                        self.image_progress.emit('2')
                                        
                                except Exception as e:
                                    print(e)
                                    pass
                        except Exception as e:
                            
                            print(e)
                            self.image_progress.emit('3')
                        
                    
                
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
                        if 'Premium' in line:
                            
                            resolution = re.findall(r'[\d]*x[\d]*',line)
                            
                            res=resolution[0] + ' (Enhanced bitrate)'
                            resolutions_list.append(res)
                            id=line[:3]
                            fps=(line[fps_index:fps_index+3])
                            print(fps)
                            self.dict_res_id_fps[res] = [id,fps]
                            self.addRes.emit(res)
                        if 'mp4' in line:
                            
                            resolution = re.findall(r'[\d]*x[\d]*',line)
                            if len(resolution) > 0:
                                if resolution[0] not in resolutions_list:
                                    res=resolution[0]
                                    resolutions_list.append(res)
                                    id=line[:3]
                                    fps=(line[fps_index:fps_index+3])
                                    print(fps)
                                    self.dict_res_id_fps[res] = [id,fps]
                                    self.addRes.emit(res)
                        
                    self.originalSelf.duration = self.originalSelf.get_youtube_video_duration(self.url)
                    name = self.originalSelf.get_youtube_video_name(self.url)
                    
                    self.originalSelf.input_file = f'{thisdir}/{name}.mp4'
                    self.originalSelf.input_file = self.originalSelf.input_file.replace('"',"")
                    self.originalSelf.main.videoName = f'{name}.mp4'
                    self.finished.emit(self.dict_res_id_fps)
                else:
                    self.progress.emit(result.stderr)
        except Exception as e:
                print(e)

#This script creates a class that takes in params like "RealESRGAN or Rife", the model for the program,  the times of upscaling, and the path of the video, and the output path
# hz
import src.return_data as return_data
import os
from src.settings import *
import glob
from threading import Thread
import src.runAI.transition_detection
from src.return_data import *
from src.messages import *
from src.discord_rpc import *
import glob
import os
from modules.commands import *
import src.thisdir
thisdir = src.thisdir.thisdir()
homedir = os.path.expanduser(r"~")
    
class interpolation(QObject):
    
    finished = pyqtSignal()
    log = pyqtSignal(str)
    
    def __init__(self,originalSelf,model,parent=None):
        self.originalSelf = originalSelf
        self.model = model
        self.main = originalSelf
        QThread.__init__(self, parent)
   
             
            

    
    def start_Render(self):
            
            
            times = self.main.times
            videopath = self.main.input_file
            outputpath = self.main.output_folder
            
            # Have to put this before otherwise it will error out ???? idk im not good at using qt.....
                    
                    
            #self.main.runLogs(videoName,times)
            start(self,self.main,self.main.render_folder,self.main.videoName,videopath,times)
            self.main.transitionDetection = src.runAI.transition_detection.TransitionDetection(self.main)
            self.main.transitionDetection.find_timestamps()
            self.main.transitionDetection.get_frame_num(times)
            self.main.endNum = 0 # This variable keeps track of the amound of zeros to fill in the output frames, this helps with pausing and resuming so rife wont overwrite the original frames.
            self.Render(self.model,times,videopath,outputpath)
            
            
            
            
            
        
            
    def Render(self,model,times,videopath,outputpath):   
            self.main.paused = False
            settings=Settings()
            input_frames = len(os.listdir(f'{self.main.render_folder}/{self.main.videoName}_temp/input_frames/'))
            if self.main.AI == 'rife-ncnn-vulkan':
                if model == 'rife-v4.6' or model == 'rife-v4':
                    os.system(f'"{settings.ModelDir}/rife/rife-ncnn-vulkan" -n {input_frames*times}  -m  {self.model} -i "{self.main.render_folder}/{self.main.videoName}_temp/input_frames/" -o "{self.main.render_folder}/{self.main.videoName}_temp/output_frames/" {return_gpu_settings(self.main)} -f %08d{self.main.settings.Image_Type}')
                else:
                    os.system(f'"{settings.ModelDir}/rife/rife-ncnn-vulkan"  -m  {self.model} -i "{self.main.render_folder}/{self.main.videoName}_temp/input_frames/" -o "{self.main.render_folder}/{self.main.videoName}_temp/output_frames/" {return_gpu_settings(self.main)} -f %08d{self.main.settings.Image_Type} ')
            if os.path.exists(f'{self.main.render_folder}/{self.main.videoName}_temp/output_frames/') == False:
                show_on_no_output_files(self.main)
            
            else:
                self.main.transitionDetection.merge_frames()
                self.log.emit("[Merging Frames]")
                self.main.output_file = end(self,self.main,self.main.render_folder,self.main.videoName,videopath,times,outputpath, self.main.videoQuality,self.main.encoder)
                
                self.finished.emit()
                
class upscale(QObject):
    finished = pyqtSignal()
    log = pyqtSignal(str)
    
    def __init__(self,originalSelf,parent=None):
        self.originalSelf = originalSelf
        self.main = originalSelf
        QThread.__init__(self, parent)
    def start_Render(self):

        start(self,self.main,self.main.render_folder,self.main.videoName,self.main.input_file,1)
        
        self.realESRGAN()
    def realESRGAN(self):
            settings = Settings()
            self.main.endNum=0
            self.main.paused=False
            img_type = self.main.settings.Image_Type.replace('.','')
            if self.main.AI == 'realesrgan-ncnn-vulkan':
                os.system(f'"{settings.ModelDir}/realesrgan/realesrgan-ncnn-vulkan" -i "{self.main.render_folder}/{self.main.videoName}_temp/input_frames" -o "{self.main.render_folder}/{self.main.videoName}_temp/output_frames" {self.main.realESRGAN_Model}{return_gpu_settings(self.main)} -f {img_type} ')
            if os.path.exists(f'{self.main.render_folder}/{self.main.videoName}_temp/output_frames/') == False:
                    show_on_no_output_files(self.main)
            else:
                    if self.main.paused == False:
                        self.log.emit("[Merging Frames]")
                        self.main.output_file = end(self,self.main,self.main.render_folder,self.main.videoName,self.main.input_file,1,self.main.output_folder, self.main.videoQuality,self.main.encoder,'upscale')
                    else:
                        pass
            self.finished.emit()