
from PyQt5 import QtWidgets, uic
import sys
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QMessageBox, QListWidget, QListWidgetItem
from PyQt5.QtGui import QTextCursor
import mainwindow
import os
from threading import *
from src.programData.settings import *
from src.programData.return_data import *
from time import sleep
import src.programData.thisdir
from src.misc.log import log
import traceback
import src.programData.return_data as return_data
import src.runAI.transition_detection as transition_detection
import os
from modules.commands import *
import math
import src.programData.checks as checks

thisdir = src.programData.thisdir.thisdir()
homedir = os.path.expanduser(r"~")
class pb2X(QObject):
    finished = pyqtSignal()
    image_progress = pyqtSignal(str)
    progress = pyqtSignal(int)
    def __init__(self,myvar,render,main,parent=None):
        
        QThread.__init__(self, parent)
        self.input_file = myvar
        self.videoName = VideoName.return_video_name(f'{self.input_file}')
        
        
        self.main = main
    def run(self):
        self.settings = Settings()
        log('Start Progressbar/Info Thread')
        try:
            while ManageFiles.isfolder(f'{self.settings.RenderDir}/{self.videoName}_temp/output_frames/') == False:
                sleep(.1) # has to refresh quickly or small files that interpolate fast do not work
            

            total_input_files = len(os.listdir(f'{self.settings.RenderDir}/{self.videoName}_temp/input_frames/'))
            total_output_files = total_input_files * self.main.times
            # fc is the total file count after interpolation
            #Could use this instead of just os.listdir
            
            
            while ManageFiles.isfolder(f'{self.settings.RenderDir}/{self.videoName}_temp/') == True:
                    if ManageFiles.isfolder(f'{self.settings.RenderDir}/{self.videoName}_temp/output_frames/') == True:
                        try:
                            if self.settings.RenderType == 'Optimized (Incremental)':
                                try:
                                    files_processed = os.listdir(f'{self.settings.RenderDir}/{self.videoName}_temp/output_frames/0/')
                                    files_processed.sort()
                                    files_processed = files_processed[-1]
                                    files_processed = files_processed.replace(self.settings.Image_Type,'')
                                    files_processed = int(files_processed)
                                    self.main.files_processed = int((len(os.listdir(f"{self.settings.RenderDir}/{self.videoName}_temp/output_frames/"))-1)/interpolation_sessions*total_output_files)
                                    
                                except:
                                    #print('i really gotta fix this')
                                    pass
                            if self.settings.RenderType == 'Optimized':
                                try:
                                    files_processed = os.listdir(f'{self.settings.RenderDir}/{self.videoName}_temp/output_frames/0/')
                                    files_processed.sort()
                                    files_processed = files_processed[-1]
                                    files_processed = files_processed.replace(self.settings.Image_Type,'')
                                    files_processed = int(files_processed)
                                    self.main.files_processed = files_processed
                                except Exception as e:
                                    self.main.files_processed = 0
                                    tb = traceback.format_exc()
                                    #print(f'{e},{tb}')

                            else:
                                files_processed = len(os.listdir(f'{self.settings.RenderDir}/{self.videoName}_temp/output_frames/0/'))
                                self.main.files_processed = files_processed
                                
                            try:
                                self.progress.emit(self.main.files_processed)
                            except Exception as e:
                                #print(e)
                                pass
                                
                            sleep(.1)
                            
                            self.main.imageDisplay=f'{self.settings.RenderDir}/{self.main.videoName}_temp/output_frames/0/{str(files_processed).zfill(8)}{self.settings.Image_Type}' # sets behind to stop corrupted jpg error
                            if self.main.imageDisplay != None:

                                try:
                                    if os.path.exists(self.main.imageDisplay):
                                        self.image_progress.emit('1')
                                        
                                        width = self.main.width()
                                        height = self.main.height()
                                        
                                        self.main.width1=int(width/1.6)
                                        self.main.height1=int(self.main.width1/self.main.aspectratio)
                                        if self.main.height1 >= height/1.6:
                                            
                                            self.main.height1=int(height/1.6)
                                            self.main.width1=int(self.main.height1/(self.main.videoheight/self.main.videowidth))
                                        try:
                                            if os.path.exists(self.main.imageDisplay):
                                                
                                                
                                                
                                                self.image_progress.emit('2')
                                                
                                        except Exception as e:
                                            traceback_info = traceback.format_exc()
                                            log(f'{e} {traceback_info}')
                                            pass
                                except Exception as e:
                                    
                                    traceback_info = traceback.format_exc()
                                    log(f'{e} {traceback_info}')
                                    self.image_progress.emit('3')
                        except Exception as e:
                                    
                                    traceback_info = traceback.format_exc()
                                    log(f'{e} {traceback_info}')
                    else:
                        pass
                        #log('No render folder exists!') 
                        #print('No render folder exists!')             
                            
                        
                    
            sleep(1)
            self.finished.emit()

        except Exception as e:
                traceback_info = traceback.format_exc()
                log(f'{e} {traceback_info}')
class downloadVideo(QObject):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(str)
    addRes = pyqtSignal(str)
    def __init__(self,originalSelf,url,parent=None):
        self.originalSelf = originalSelf
        self.url = url
        QThread.__init__(self, parent)
    def run(self):
        log('INFO: Downloading Video')
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
                            #print(fps_index)
                            break
                    for line in reversed(stdout_lines):
                        if 'Premium' in line:
                            
                            resolution = re.findall(r'[\d]*x[\d]*',line)
                            
                            res=resolution[0] + ' (Enhanced bitrate)'
                            resolutions_list.append(res)
                            id=line[:3]
                            fps=(line[fps_index:fps_index+3])
                            #print(fps)
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
                                    #print(fps)
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
                traceback_info = traceback.format_exc()
                log(f'{e} {traceback_info}')

#This script creates a class that takes in params like "RealESRGAN or Rife", the model for the program,  the times of upscaling, and the path of the video, and the output path
# hz


def stdlog(stdout,stderr):
    
    stdout_str = stdout.decode()
    stderr_str = stderr.decode()
    log("Standard Output:")
    log(stdout_str[:400])
    log("\nStandard Error:")
    log(stderr_str[:400])
          
#functions that ALL ncnn AI use
def render(self,command):
    self.main.renderAI = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)    
    stdout, stderr = self.main.renderAI.communicate()
    stdlog(stdout,stderr)
    
def optimized_render(self,command):
    log(f'INFO: Running AI: {command}')

    self.renderCommand = command
    global interpolation_sessions
    interpolation_sessions = int(ceildiv(self.main.frame_count,self.main.frame_increments_of_interpolation))
    self.main.interpolation_sessions = interpolation_sessions 
    #print(interpolation_sessions)
    fc_thread = Thread(target=lambda: frameCountThread(self))
    fc_thread.start()
    sleep(1)

    render(self,command)

    fc_thread.join()


def calculateFrameIncrements(self):
    
    if self.main.settings.FrameIncrementsMode == 'Manual':
        frame_increments_of_interpolation = self.main.settings.FrameIncrements
        return int(frame_increments_of_interpolation)
    elif self.main.settings.FrameIncrementsMode == 'Automatic':
       
        width,height=VideoName.return_video_resolution(self.main.input_file)
        
       
        cap = cv2.VideoCapture(self.main.input_file)
        if not cap.isOpened():
            print("Error opening video file")
            log('ERROR: Could not open video file')
            return

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fps = cap.get(cv2.CAP_PROP_FPS)

        duration_seconds = frame_count // fps
 
        #divisor = len(str(int(width))) 
        frame_increments_of_interpolation = int((duration_seconds))
        print(frame_increments_of_interpolation)
        log(f'Frames per video increment: {frame_increments_of_interpolation}')
        if int(frame_increments_of_interpolation) <=10:
            frame_increments_of_interpolation = 100
        return int(frame_increments_of_interpolation)

def calculateVRAM(self):
    
    width,height = return_data.VideoName.return_video_resolution(self.main.input_file)
    if int(height) > int(self.main.settings.UHDResCutOff): #check if resolution is greater than cutoff, if it is set to 2 if threading val is greater than or equal to 4
            if self.main.settings.VRAM >=4:
                vram=2
            else:
                vram=1
    else:
         vram = self.main.settings.VRAM
    return vram


    
def frameCountThread(self):
    log('INFO: Starting Render Thread')
    iteration = 0
    increment=1
    
    try:
        width,height = VideoName.return_video_resolution(self.main.input_file)
        width = int(width)
        height = int(height)
        self.main.vid_resolution = f'{width}x{height}'
        encoder = return_data.returnCodec(self.main.settings.Encoder)
        with open(f'{self.main.settings.RenderDir}/{self.main.videoName}_temp/output_frames/videos.txt', 'w') as f:
            for m in range(interpolation_sessions):
                f.write(f"file '{self.main.settings.RenderDir}/{self.main.videoName}_temp/output_frames/{interpolation_sessions-m}.{return_data.returnContainer(encoder)}'\n")
        if encoder != 'copy':
            vf = f'-vf "scale=w={width}:h={height},setsar=1"'
        else:
            vf=''
        transitionDetectionClass = transition_detection.TransitionDetection(self.main)
        while True:
            
            try:
                if len(os.listdir(f"{self.main.settings.RenderDir}/{self.main.videoName}_temp/output_frames/0/")) >= self.main.frame_increments_of_interpolation or iteration == interpolation_sessions-1:
                    j=1
                    
                    if iteration == interpolation_sessions-1:
                        total_frames_rendered =  abs((interpolation_sessions-1)*self.main.frame_increments_of_interpolation - self.main.frame_count)
                        #total_frames_rendered = frame_count+frame_increments_of_interpolation 
                        while j <= total_frames_rendered:
                            if os.path.isfile(f'{self.main.settings.RenderDir}/{self.main.videoName}_temp/output_frames/0/{str(increment).zfill(8)}{self.main.settings.Image_Type}'):#check if the file exists, prevents rendering issuess
                                    
                                    increment+=1
                                    j+=1

                            else:
                                sleep(.1)
                    else:
                        #Sadly i need this unoptimized check here, otherwise frames can get skipped, i tried my best
                        while j <= self.main.frame_increments_of_interpolation:
                            if os.path.isfile(f'{self.main.settings.RenderDir}/{self.main.videoName}_temp/output_frames/0/{str(increment).zfill(8)}{self.main.settings.Image_Type}'):#check if the file exists, prevents rendering issuess

                                    increment+=1
                                    j+=1
                                
                            else:
                                sleep(.1)
                    transitionDetectionClass.merge_frames()
                    os.system(f'{thisdir}/bin/ffmpeg -start_number {self.main.frame_increments_of_interpolation*iteration} -framerate {self.main.fps*self.main.times} -i "{self.main.settings.RenderDir}/{self.main.videoName}_temp/output_frames/0/%08d{self.main.settings.Image_Type}" {vf}  -frames:v  {self.main.frame_increments_of_interpolation} -c:v {return_data.returnCodec(self.main.settings.Encoder)} {returnCRFFactor(self.main.settings.videoQuality,self.main.settings.Encoder)}  -pix_fmt yuv420p  "{self.main.settings.RenderDir}/{self.main.videoName}_temp/output_frames/{interpolation_sessions-iteration}.{return_data.returnContainer(self.main.settings.Encoder)}"  -y')
                    iteration+=1
                    if iteration == interpolation_sessions:
                        break
                    for i in range(self.main.frame_increments_of_interpolation):# removes previous frames, takes the most time (optimize this?)
                        os.system(f'rm -rf "{self.main.settings.RenderDir}/{self.main.videoName}_temp/output_frames/0/{str(i+((iteration-1)*self.main.frame_increments_of_interpolation)).zfill(8)}{self.main.settings.Image_Type}"')
                    
                else:
                    sleep(0.1)
            except Exception as e:
                traceback_info = traceback.format_exc()
                log(f'{e} {traceback_info}')
    except Exception as e:
        traceb = traceback.format_exc()
        log(f'{str(e)}, {traceb}')




class interpolation(QObject):
    
    finished = pyqtSignal()
    log = pyqtSignal(str)
    removelog = pyqtSignal(str)
    def __init__(self,originalSelf,model,parent=None):
        self.originalSelf = originalSelf
        self.model = model
        self.main = originalSelf
        QThread.__init__(self, parent)
   
    def finishRenderSetup(self): #3rd and final call, called from interpolate.py
            
            extractFramesAndAudio(self,self.main,self.main.settings.RenderDir,self.main.videoName,self.main.input_file,self.main.times)
            
            # run transition detection start
            if self.main.settings.SceneChangeDetectionMode == 'Enabled' and self.main.settings.Encoder != 'Lossless':
                self.log.emit('Detecting Transitions')
                if self.main.AI == 'rife-ncnn-vulkan':
                    if 'v4' in self.model:
                         self.main.times=(self.main.ui.FPSTo.value()/self.main.ui.FPSFrom.value())
                print(self.main.times)
                self.main.transitionDetection = src.runAI.transition_detection.TransitionDetection(self.main)
                self.main.transitionDetection.find_timestamps()
                self.main.transitionDetection.get_frame_num(self.main.times)
                divisor=(self.main.times/2)
                try:
                    self.log.emit(f'Transitions detected: {str(int(len(os.listdir(f"{self.main.settings.RenderDir}/{self.main.videoName}_temp/transitions/"))//divisor))}')
                except:
                    self.log.emit(f'Transitions detected: 0')
            self.Render(self.model,self.main.times,self.main.input_file,self.main.output_folder)
    import math
    def Render(self,model,times,videopath,outputpath):
            
                self.main.paused = False
                settings=Settings()
                self.log.emit(f'Starting {str(round(self.main.times,1))[:3]}X Render')
                self.log.emit(f'Model: {self.main.ui.Rife_Model.currentText()}')
                self.input_frames = len(os.listdir(f'{settings.RenderDir}/{self.main.videoName}_temp/input_frames/'))
                
                self.main.frame_count = self.input_frames * self.main.times # frame count of video multiplied by times 
                
                self.main.frame_increments_of_interpolation = int(calculateFrameIncrements(self))
                vram = int(calculateVRAM(self))
                
                
                if self.main.AI == 'rife-ncnn-vulkan':
                    
                    if int(self.main.videoheight) > int(settings.UHDResCutOff):
                        uhd_mode = '-u'
                    else:
                        uhd_mode = ''
                    
                    command = [
    f'{settings.ModelDir}/rife/rife-ncnn-vulkan',
    '-m', self.model,
    '-i', f'{settings.RenderDir}/{self.main.videoName}_temp/input_frames/',
    '-o', f'{settings.RenderDir}/{self.main.videoName}_temp/output_frames/0/',
    '-j', f'{math.ceil(vram/4)}:{vram}:{math.ceil(vram/4)+1}',
    '-f', f'%08d{self.main.settings.Image_Type}',
    '-g', f'{self.main.ui.gpuIDSpinBox.value()}'
    f'{uhd_mode}']
    
                    if 'v4' in model:
                         command.append('-n')
                         command.append(str(math.ceil(self.input_frames * times)))
                         print(command)
                    if settings.RenderType == 'Optimized' and self.main.frame_count > self.main.frame_increments_of_interpolation and self.main.frame_increments_of_interpolation > 0:
                        

                        optimized_render(self,command)
                    
                    else:
                        render(self,command)
                    

                if self.main.AI == 'ifrnet-ncnn-vulkan':               
                    
                    command = [
f'{settings.ModelDir}/ifrnet/ifrnet-ncnn-vulkan',

    '-i', f'{settings.RenderDir}/{self.main.videoName}_temp/input_frames/',
    '-o', f'{settings.RenderDir}/{self.main.videoName}_temp/output_frames/0/',
    '-j', f'1:{vram}:2',
    '-f', f'%08d{self.main.settings.Image_Type}',
    '-m', f'{settings.ModelDir}ifrnet/{self.main.ui.Rife_Model.currentText()}',
    '-g', f'{self.main.ui.gpuIDSpinBox.value()}'
]
                    
                    if settings.RenderType == 'Optimized' and self.main.frame_count > self.main.frame_increments_of_interpolation and self.main.frame_increments_of_interpolation > 0:
                        

                        optimized_render(self,command)
                    
                    else:
                        render(self,command)
            
                if os.path.exists(f'{settings.RenderDir}/{self.main.videoName}_temp/output_frames/') == False:
                    show_on_no_output_files(self.main)
                
                else:
                    if settings.SceneChangeDetectionMode == 'Enabled':
                        self.main.transitionDetection.merge_frames()
                    if settings.RenderType != 'Optimized':
                        self.log.emit("[Merging Frames]")
                    self.main.output_file = end(self,self.main,settings.RenderDir,self.main.videoName,videopath,times,outputpath, self.main.videoQuality,self.main.encoder)
                    
                self.finished.emit()
            
                
                
class upscale(QObject):
    finished = pyqtSignal()
    log = pyqtSignal(str)
    removelog = pyqtSignal(str)
    def __init__(self,originalSelf,parent=None):
        self.originalSelf = originalSelf
        self.main = originalSelf
        QThread.__init__(self, parent)
    def finishRenderSetup(self): #3rd and final call, called from upscale.py

        extractFramesAndAudio(self,self.main,self.main.settings.RenderDir,self.main.videoName,self.main.input_file,1)
        
        self.realESRGAN()

    def realESRGAN(self):
        
            settings = Settings()
            self.main.endNum=0
            self.main.paused=False

            self.main.frame_increments_of_interpolation = calculateFrameIncrements(self)
            img_type = self.main.settings.Image_Type.replace('.','')
            self.input_frames = len(os.listdir(f'{settings.RenderDir}/{self.main.videoName}_temp/input_frames/'))
            self.main.frame_count = self.input_frames
            
            if self.main.AI == 'realesrgan-ncnn-vulkan':
                command = [
    f'{settings.ModelDir}/realesrgan/realesrgan-ncnn-vulkan',
    '-i', f'{settings.RenderDir}/{self.main.videoName}_temp/input_frames',
    '-o', f'{settings.RenderDir}/{self.main.videoName}_temp/output_frames/0/',
    '-j', f'1:{settings.VRAM}:2',
    '-f', str(img_type)
]
                for i in self.main.realESRGAN_Model.split(' '):
                        
                        command.append(i)
                        print(command)
                

            if self.main.AI == 'waifu2x-ncnn-vulkan':
                command = [
    f'{settings.ModelDir}/waifu2x/waifu2x-ncnn-vulkan',
    '-i', f'{settings.RenderDir}/{self.main.videoName}_temp/input_frames',
    '-o', f'{settings.RenderDir}/{self.main.videoName}_temp/output_frames/0/',
    '-s', str(int(self.main.ui.Rife_Times.currentText()[0])),
    '-n', str(self.main.ui.denoiseLevelSpinBox.value()),
    '-j', f'1:{settings.VRAM}:2',
    '-f', str(img_type),
    '-m', f'{settings.ModelDir}waifu2x/models-{self.main.ui.Rife_Model.currentText()}',
    '-g', f'{self.main.ui.gpuIDSpinBox.value()}'
]
            if self.main.AI == 'realcugan-ncnn-vulkan':
                command = [
    f'{settings.ModelDir}/realcugan/realcugan-ncnn-vulkan',
    '-i', f'{settings.RenderDir}/{self.main.videoName}_temp/input_frames',
    '-o', f'{settings.RenderDir}/{self.main.videoName}_temp/output_frames/0/',
    '-s', str(int(self.main.ui.Rife_Times.currentText()[0])),
    '-n', str(self.main.ui.denoiseLevelSpinBox.value()),
    '-j', f'1:{settings.VRAM}:2',
    '-f', str(img_type),
    '-m', f'{settings.ModelDir}realcugan/{self.main.ui.Rife_Model.currentText()}',
    '-g', f'{self.main.ui.gpuIDSpinBox.value()}'
]
            if self.main.AI == 'realsr-ncnn-vulkan':
                command = [
    f'{settings.ModelDir}realsr/realsr-ncnn-vulkan',
    '-i', f'{settings.RenderDir}/{self.main.videoName}_temp/input_frames',
    '-o', f'{settings.RenderDir}/{self.main.videoName}_temp/output_frames/0/',
    '-j', f'1:{settings.VRAM}:2',
    '-f', str(img_type),
    '-m', f'{settings.ModelDir}realsr/models-{self.main.ui.Rife_Model.currentText()}',
    '-g', f'{self.main.ui.gpuIDSpinBox.value()}'
]
            if settings.RenderType == 'Optimized' and self.main.frame_count > self.main.frame_increments_of_interpolation and self.main.frame_increments_of_interpolation > 0:
                
                optimized_render(self,command)
            else:
                render(self,command)
            if os.path.exists(f'{settings.RenderDir}/{self.main.videoName}_temp/output_frames/') == False:
                    show_on_no_output_files(self.main)
            else:
                    if self.main.paused == False:
                        if settings.RenderType != 'Optimized':
                            self.log.emit("[Merging Frames]")
                            log('INFO: Merging Frames')
                        self.main.output_file = end(self,self.main,settings.RenderDir,self.main.videoName,self.main.input_file,1,self.main.output_folder, self.main.videoQuality,self.main.encoder,'upscale')
                    else:
                        pass
            self.finished.emit()
