
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
from src.log import log
import traceback
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
                                print('i really gotta fix this')
                                pass
                        if self.settings.RenderType == 'Optimized':
                            try:
                                files_processed = os.listdir(f'{self.settings.RenderDir}/{self.videoName}_temp/output_frames/0/')
                                files_processed.sort()
                                files_processed = files_processed[-1]
                                files_processed = files_processed.replace(self.settings.Image_Type,'')
                                files_processed = int(files_processed)
                                self.main.files_processed = files_processed
                            except:
                                pass
                                #print('i really gotta fix this')

                        else:
                            files_processed = len(os.listdir(f'{self.settings.RenderDir}/{self.videoName}_temp/output_frames/0/'))
                            self.main.files_processed = files_processed
                            
                        try:
                            self.progress.emit(self.main.files_processed)
                        except Exception as e:
                            print(e)
                            
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
import src.return_data as return_data
import os
from src.settings import *
from threading import Thread
import src.runAI.transition_detection as transition_detection
from src.return_data import *
from src.messages import *
from src.discord_rpc import *
import os
from modules.commands import *
import src.thisdir
import math
from PIL import Image
thisdir = src.thisdir.thisdir()
homedir = os.path.expanduser(r"~")
def ffmpeg_preset(self):
     
     vq = int(self.main.settings.videoQuality)
     if vq < 18:
          return '-preset slow'
     
          



    
def frameCountThread(self):#in theory, this function will keep moving out frames into a different folder based on a number of how much the video should be split up too, this can severly lower size of interpolation
    global iteration
    iteration = 0
    increment=1
    with open(f'{self.main.settings.RenderDir}/{self.main.videoName}_temp/output_frames/videos.txt', 'w') as f:
        for m in range(interpolation_sessions):
            f.write(f'file {interpolation_sessions-m}.mp4\n')
    
    while True:
        
        try:
            if len(os.listdir(f"{self.main.settings.RenderDir}/{self.main.videoName}_temp/output_frames/0/")) >= frame_increments_of_interpolation or iteration == interpolation_sessions-1:
                j=1
                
                if iteration == interpolation_sessions-1:
                    total_frames_rendered =  abs((interpolation_sessions-1)*frame_increments_of_interpolation - frame_count)
                    #total_frames_rendered = frame_count+frame_increments_of_interpolation 
                    while j <= total_frames_rendered:
                        if os.path.isfile(f'{self.main.settings.RenderDir}/{self.main.videoName}_temp/output_frames/0/{str(increment).zfill(8)}{self.main.settings.Image_Type}'):#check if the file exists, prevents rendering issuess
                                
                                increment+=1
                                j+=1
                            

                            
                        else:
                            sleep(.1)
                else:
                    #Sadly i need this unoptimized check here, otherwise frames can get skipped, i tried my best
                    while j <= frame_increments_of_interpolation:
                        if os.path.isfile(f'{self.main.settings.RenderDir}/{self.main.videoName}_temp/output_frames/0/{str(increment).zfill(8)}{self.main.settings.Image_Type}'):#check if the file exists, prevents rendering issuess
                            
                            
                                increment+=1
                                j+=1
                            
                        else:
                            sleep(.1)
                transitionDetectionClass.merge_frames()
                os.system(f'{thisdir}/bin/ffmpeg -start_number {frame_increments_of_interpolation*iteration} -framerate {self.main.fps*self.main.times} -i "{self.main.settings.RenderDir}/{self.main.videoName}_temp/output_frames/0/%08d{self.main.settings.Image_Type}" -frames:v  {frame_increments_of_interpolation} -c:v libx{self.main.settings.Encoder} -crf {self.main.settings.videoQuality}  -pix_fmt yuv420p  "{self.main.settings.RenderDir}/{self.main.videoName}_temp/output_frames/{interpolation_sessions-iteration}.mp4"  -y')
                iteration+=1
                if iteration == interpolation_sessions:
                    break
                for i in range(frame_increments_of_interpolation):# removes previous frames, takes the most time (optimize this?)
                       os.system(f'rm -rf "{self.main.settings.RenderDir}/{self.main.videoName}_temp/output_frames/0/{str(i+((iteration-1)*frame_increments_of_interpolation)).zfill(8)}{self.main.settings.Image_Type}"')
                '''os.chdir(f'{self.main.settings.RenderDir}/{self.main.videoName}_temp/output_frames/0/')

                os.system(f'rm -rf {{{str((iteration*frame_increments_of_interpolation)).zfill(8)}..{str((iteration*frame_increments_of_interpolation+frame_increments_of_interpolation)).zfill(8)}}}{self.main.settings.Image_Type}')
                
                os.chdir(f'{thisdir}')'''
            else:
                sleep(0.1)
        except Exception as e:
            print(e)
            log(str(e))

def AI(self,command):
    global transitionDetectionClass
    transitionDetectionClass = transition_detection.TransitionDetection(self.main)
    
    self.renderCommand = command
    global interpolation_sessions
    interpolation_sessions = ceildiv(frame_count,frame_increments_of_interpolation)
    self.main.interpolation_sessions = interpolation_sessions
    fc_thread = Thread(target=lambda: frameCountThread(self))
    fc_thread.start()
    sleep(1)
    
    
    self.main.renderAI = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    stdout, stderr = self.main.renderAI.communicate()

    # Decode the byte strings to get text output
    stdout_str = stdout.decode()
    stderr_str = stderr.decode()

    # Print or handle stdout and stderr as needed
    print("Standard Output:")
    print(stdout_str)

    print("\nStandard Error:")
    print(stderr_str)
    print('\n\n\n\n')
    #'./rife/rife-ncnn-vulkan -m rife/rife-v4.6 -i input_frames -o output_frames/0'
    #merge all videos created here
    fc_thread.join()

def AI_Incremental(self,command):
    global transitionDetectionClass
    transitionDetectionClass = transition_detection.TransitionDetection(self.main)
    
    
    global interpolation_sessions
    interpolation_sessions = ceildiv(frame_count,frame_increments_of_interpolation)
    
    '''fc_thread = Thread(target=lambda: frameCountThread(self))
    fc_thread.start()'''
    sleep(1)
    print('\n\n\n\n')
    #this is only triggered on interpolation, because self.model is set to '' on upscale
    
    
    for i in range(len(os.listdir(f'{self.main.render_folder}/{self.main.videoName}_temp/input_frames/'))):
        try:
            os.mkdir(f'{self.main.render_folder}/{self.main.videoName}_temp/output_frames/0/')
        except:
            pass
        
        command = command.replace(f'input_frames/{i-1}/', f'input_frames/{i}/')
        if i >= (len(os.listdir(f'{self.main.render_folder}/{self.main.videoName}_temp/input_frames/'))) -1:
            command = command.replace(f'-n {frame_increments_of_interpolation}', f'-n {len(os.listdir(f"{self.main.render_folder}/{self.main.videoName}_temp/input_frames/{i}/"))}')
        os.system(command)
        #transitionDetectionClass.merge_frames(i)
        os.system(f'{thisdir}/bin/ffmpeg  -framerate {self.main.fps*self.main.times} -i "{self.main.settings.RenderDir}/{self.main.videoName}_temp/output_frames/0/%08d{self.main.settings.Image_Type}"  -c:v libx{self.main.settings.Encoder} -crf {self.main.settings.videoQuality}  -pix_fmt yuv420p  "{self.main.settings.RenderDir}/{self.main.videoName}_temp/output_frames/{i}.mp4"  -y')#replace png with image type
        
        with open(f'{self.main.settings.RenderDir}/{self.main.videoName}_temp/output_frames/videos.txt', 'a') as f:
            f.write(f'file {i}.mp4\n')
        shutil.rmtree(f'{self.main.settings.RenderDir}/{self.main.videoName}_temp/output_frames/0/')
    #'./rife/rife-ncnn-vulkan -m rife/rife-v4.6 -i input_frames -o output_frames/0'
    #merge all videos created here
    ''' fc_thread.join()'''
    
                #print('file 0 succsessfully made.')

class interpolation(QObject):
    
    finished = pyqtSignal()
    log = pyqtSignal(str)
    removelog = pyqtSignal(str)
    def __init__(self,originalSelf,model,parent=None):
        self.originalSelf = originalSelf
        self.model = model
        self.main = originalSelf
        QThread.__init__(self, parent)
   
             
            

    
    def start_Render(self):
            
        try:    
            times = self.main.times
            videopath = self.main.input_file
            outputpath = self.main.output_folder
            
            # Have to put this before otherwise it will error out ???? idk im not good at using qt.....
            
                    
            #self.main.runLogs(videoName,times)
            start(self,self.main,self.main.render_folder,self.main.videoName,videopath,times)
            if self.main.settings.SceneChangeDetectionMode == 'Enabled':
                self.main.transitionDetection = src.runAI.transition_detection.TransitionDetection(self.main)
                self.main.transitionDetection.find_timestamps()
                self.main.transitionDetection.get_frame_num(times)
            self.main.endNum = 0 # This variable keeps track of the amound of zeros to fill in the output frames, this helps with pausing and resuming so rife wont overwrite the original frames.
            self.Render(self.model,times,videopath,outputpath)
        except Exception as e:
                traceback_info = traceback.format_exc()
                log(f'{e} {traceback_info}')
                self.main.showDialogBox(e) 
                   
            
            
            
            
        
            
    def Render(self,model,times,videopath,outputpath):  
            try: 
                self.main.paused = False
                settings=Settings()
                
                self.input_frames = len(os.listdir(f'{self.main.render_folder}/{self.main.videoName}_temp/input_frames/'))
                global frame_count
                frame_count = self.input_frames * self.main.times # frame count of video multiplied by times 
                global frame_increments_of_interpolation
                if self.main.settings.FrameIncrementsMode == 'Manual':
                    frame_increments_of_interpolation = self.main.settings.FrameIncrements
                elif self.main.settings.FrameIncrementsMode == 'Automatic':
                    resolution = VideoName.return_video_resolution(self.main.input_file)
                    try:
                        frame_increments_of_interpolation = int(100*int(self.main.settings.VRAM)/(round(int(resolution[0])/1000)))
                    except:
                         frame_increments_of_interpolation = int(100*int(self.main.settings.VRAM))
                    frame_increments_of_interpolation = int(frame_increments_of_interpolation)
                self.main.frame_increments_of_interpolation = frame_increments_of_interpolation
                if self.main.AI == 'rife-ncnn-vulkan':
                    if int(settings.VRAM) > 1: vram = int(int(settings.VRAM)/2)
                    else:vram=1
                    width,height = return_data.VideoName.return_video_resolution(self.main.input_file)
                    if int(width) > 3840 or int(height) > 2160:
                            vram=1
                    if 'v4' in model:
                        command = [
    f'{settings.ModelDir}/rife/rife-ncnn-vulkan',
    '-n', str(self.input_frames * times),
    '-m', self.model,
    '-i', f'{self.main.render_folder}/{self.main.videoName}_temp/input_frames/',
    '-o', f'{self.main.render_folder}/{self.main.videoName}_temp/output_frames/0/',
    '-j', f'1:{vram}:2',
    '-f', f'%08d{self.main.settings.Image_Type}']
                        if settings.RenderType == 'Optimized (Incremental)' and frame_count > frame_increments_of_interpolation and frame_increments_of_interpolation > 0:
                            AI_Incremental(self,f'"{settings.ModelDir}/rife/rife-ncnn-vulkan" -n {frame_increments_of_interpolation}  -m  {self.model} -i "{self.main.render_folder}/{self.main.videoName}_temp/input_frames/0/" -o "{self.main.render_folder}/{self.main.videoName}_temp/output_frames/0/" {return_gpu_settings(self.main)} -f %08d{self.main.settings.Image_Type}')
                        if settings.RenderType == 'Optimized' and frame_count > frame_increments_of_interpolation and frame_increments_of_interpolation > 0:
                            

                            AI(self,command)
                        
                        else:
                            self.main.renderAI = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            stdout, stderr = self.main.renderAI.communicate()

                            # Decode the byte strings to get text output
                            stdout_str = stdout.decode()
                            stderr_str = stderr.decode()

                            # Print or handle stdout and stderr as needed
                            print("Standard Output:")
                            print(stdout_str)

                            print("\nStandard Error:")
                            print(stderr_str)
                    else:
                        if int(settings.VRAM) > 1: vram = int(int(settings.VRAM)/2)
                        else:vram=1
                        width,height = return_data.VideoName.return_video_resolution(self.main.input_file)
                        if int(width) > 3840 or int(height) > 2160:
                            vram=1
                        command = [
    f'{settings.ModelDir}/rife/rife-ncnn-vulkan',
    '-m', self.model,
    '-i', f'{self.main.render_folder}/{self.main.videoName}_temp/input_frames/',
    '-o', f'{self.main.render_folder}/{self.main.videoName}_temp/output_frames/0/',
    '-j', f'1:{vram}:2',
    '-f', f'%08d{self.main.settings.Image_Type}'
]
                        if settings.RenderType == 'Optimized (Incremental)':
                            AI_Incremental(self,f'"{settings.ModelDir}/rife/rife-ncnn-vulkan"  -m  {self.model} -i "{self.main.render_folder}/{self.main.videoName}_temp/input_frames/" -o "{self.main.render_folder}/{self.main.videoName}_temp/output_frames/0/" {return_gpu_settings(self.main)} -f %08d{self.main.settings.Image_Type} ')
                        if settings.RenderType == 'Optimized' and frame_count > frame_increments_of_interpolation and frame_increments_of_interpolation > 0:
                            
                            AI(self,command)
                        else:
                            self.main.renderAI = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            stdout, stderr = self.main.renderAI.communicate()

                            # Decode the byte strings to get text output
                            stdout_str = stdout.decode()
                            stderr_str = stderr.decode()

                            # Print or handle stdout and stderr as needed
                            print("Standard Output:")
                            print(stdout_str)

                            print("\nStandard Error:")
                            print(stderr_str)

                if self.main.AI == 'ifrnet-ncnn-vulkan':               
                    if int(settings.VRAM) > 1: vram = int(int(settings.VRAM)/2)
                    else:vram=1
                    width,height = return_data.VideoName.return_video_resolution(self.main.input_file)
                    if int(width) > 3840 or int(height) > 2160:
                            vram=1
                    command = [
f'{settings.ModelDir}/ifrnet/ifrnet-ncnn-vulkan',

    '-i', f'{self.main.render_folder}/{self.main.videoName}_temp/input_frames/',
    '-o', f'{self.main.render_folder}/{self.main.videoName}_temp/output_frames/0/',
    '-j', f'1:{vram}:2',
    '-f', f'%08d{self.main.settings.Image_Type}',
    '-m', f'{settings.ModelDir}ifrnet/{self.main.ui.Rife_Model.currentText()}'
]
                    print(command)
                    
                    if settings.RenderType == 'Optimized (Incremental)' and frame_count > frame_increments_of_interpolation and frame_increments_of_interpolation > 0:
                        AI_Incremental(self,f'"{settings.ModelDir}/ifrnet/ifrnet-ncnn-vulkan" -n {frame_increments_of_interpolation}  -m  {self.model} -i "{self.main.render_folder}/{self.main.videoName}_temp/input_frames/0/" -o "{self.main.render_folder}/{self.main.videoName}_temp/output_frames/0/" {return_gpu_settings(self.main)} -f %08d{self.main.settings.Image_Type}')
                    if settings.RenderType == 'Optimized' and frame_count > frame_increments_of_interpolation and frame_increments_of_interpolation > 0:
                        

                        AI(self,command)
                    
                    else:
                        self.main.renderAI = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        stdout, stderr = self.main.renderAI.communicate()

                        # Decode the byte strings to get text output
                        stdout_str = stdout.decode()
                        stderr_str = stderr.decode()

                        # Print or handle stdout and stderr as needed
                        print("Standard Output:")
                        print(stdout_str)

                        print("\nStandard Error:")
                        print(stderr_str)
            
                if os.path.exists(f'{self.main.render_folder}/{self.main.videoName}_temp/output_frames/') == False:
                    show_on_no_output_files(self.main)
                
                else:
                    if settings.SceneChangeDetectionMode == 'Enabled':
                        self.main.transitionDetection.merge_frames()
                    if settings.RenderType != 'Optimized':
                        self.log.emit("[Merging Frames]")
                    self.main.output_file = end(self,self.main,self.main.render_folder,self.main.videoName,videopath,times,outputpath, self.main.videoQuality,self.main.encoder)
                    
                    self.finished.emit()
            except Exception as e:
                traceback_info = traceback.format_exc()
                log(f'{e} {traceback_info}')
                self.main.showDialogBox(e)   
                
                
class upscale(QObject):
    finished = pyqtSignal()
    log = pyqtSignal(str)
    removelog = pyqtSignal(str)
    def __init__(self,originalSelf,parent=None):
        self.originalSelf = originalSelf
        self.main = originalSelf
        QThread.__init__(self, parent)
    def start_Render(self):

        start(self,self.main,self.main.render_folder,self.main.videoName,self.main.input_file,1)
        
        self.realESRGAN()
    def realESRGAN(self):
        try:
            settings = Settings()
            self.main.endNum=0
            self.main.paused=False
            global frame_count
            global frame_increments_of_interpolation
            if self.main.settings.FrameIncrementsMode == 'Manual':
                frame_increments_of_interpolation = self.main.settings.FrameIncrements
            elif self.main.settings.FrameIncrementsMode == 'Automatic':
                resolution = VideoName.return_video_resolution(self.main.input_file)
                cap = cv2.VideoCapture(self.main.input_file)
                if not cap.isOpened():
                    print("Error opening video file")
                    return

                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                fps = cap.get(cv2.CAP_PROP_FPS)

                duration_seconds = frame_count / fps

                duration_minutes = int(duration_seconds // 60)
                duration_seconds = int(duration_seconds % 60)
                frame_increments_of_interpolation = int(duration_seconds*int(self.main.settings.VRAM))
                
                frame_increments_of_interpolation = int(frame_increments_of_interpolation)
            self.main.frame_increments_of_interpolation = frame_increments_of_interpolation
            img_type = self.main.settings.Image_Type.replace('.','')
            self.input_frames = len(os.listdir(f'{self.main.render_folder}/{self.main.videoName}_temp/input_frames/'))
            frame_count = self.input_frames
            
            if self.main.AI == 'realesrgan-ncnn-vulkan':
                command = [
    f'{settings.ModelDir}/realesrgan/realesrgan-ncnn-vulkan',
    '-i', f'{self.main.render_folder}/{self.main.videoName}_temp/input_frames',
    '-o', f'{self.main.render_folder}/{self.main.videoName}_temp/output_frames/0/',
    '-j', f'1:{settings.VRAM}:2',
    '-f', str(img_type)
]
                for i in self.main.realESRGAN_Model.split(' '):
                        
                        command.append(i)
                        print(command)
                if settings.RenderType == 'Optimized (Incremental)' and frame_count > frame_increments_of_interpolation and frame_increments_of_interpolation > 0:
                    AI_Incremental(self,f'"{settings.ModelDir}/realesrgan/realesrgan-ncnn-vulkan" -i "{self.main.render_folder}/{self.main.videoName}_temp/input_frames/0/" -o "{self.main.render_folder}/{self.main.videoName}_temp/output_frames/0/" {self.main.realESRGAN_Model} {return_gpu_settings(self.main)} -f {img_type} ')
                if settings.RenderType == 'Optimized' and frame_count > frame_increments_of_interpolation and frame_increments_of_interpolation > 0:
                    
                    
                    
                    AI(self,command)
                else:
                    self.main.renderAI = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr = self.main.renderAI.communicate()

                    # Decode the byte strings to get text output
                    stdout_str = stdout.decode()
                    stderr_str = stderr.decode()

                    # Print or handle stdout and stderr as needed
                    print("Standard Output:")
                    print(stdout_str)

                    print("\nStandard Error:")
                    print(stderr_str)
    
            if self.main.AI == 'waifu2x-ncnn-vulkan':
                command = [
    f'{settings.ModelDir}/waifu2x/waifu2x-ncnn-vulkan',
    '-i', f'{self.main.render_folder}/{self.main.videoName}_temp/input_frames',
    '-o', f'{self.main.render_folder}/{self.main.videoName}_temp/output_frames/0/',
    '-s', str(int(self.main.ui.Rife_Times.currentText()[0])),
    '-n', str(self.main.ui.denoiseLevelSpinBox.value()),
    '-j', f'{settings.VRAM}:{settings.VRAM}:{settings.VRAM}',
    '-f', str(img_type),
    '-m', f'{settings.ModelDir}waifu2x/models-{self.main.ui.Rife_Model.currentText()}',
]
                print(command)
                if settings.RenderType == 'Optimized (Incremental)' and frame_count > frame_increments_of_interpolation and frame_increments_of_interpolation > 0:
                    AI_Incremental(self,f'"{settings.ModelDir}/waifu2x/waifu2x-ncnn-vulkan" -i "{self.main.render_folder}/{self.main.videoName}_temp/input_frames/0/" -o "{self.main.render_folder}/{self.main.videoName}_temp/output_frames/0/" -s {int(self.main.ui.Rife_Times.currentText()[0])} {return_gpu_settings(self.main)} -f {img_type} ')
                if settings.RenderType == 'Optimized' and frame_count > frame_increments_of_interpolation and frame_increments_of_interpolation > 0:
                    
                    AI(self,command)
                else:
                    self.main.renderAI = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr = self.main.renderAI.communicate()

                    # Decode the byte strings to get text output
                    stdout_str = stdout.decode()
                    stderr_str = stderr.decode()

                    # Print or handle stdout and stderr as needed
                    print("Standard Output:")
                    print(stdout_str)

                    print("\nStandard Error:")
                    print(stderr_str)
            if os.path.exists(f'{self.main.render_folder}/{self.main.videoName}_temp/output_frames/') == False:
                    show_on_no_output_files(self.main)
            else:
                    if self.main.paused == False:
                        self.log.emit("[Merging Frames]")
                        self.main.output_file = end(self,self.main,self.main.render_folder,self.main.videoName,self.main.input_file,1,self.main.output_folder, self.main.videoQuality,self.main.encoder,'upscale')
                    else:
                        pass
            self.finished.emit()
        except Exception as e:
            traceback_info = traceback.format_exc()
            log(f'{e} {traceback_info}')
            self.main.showDialogBox(e)   
        
