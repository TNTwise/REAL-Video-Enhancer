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

thisdir= os.getcwd()
homedir = os.path.expanduser(r"~")
def renderRealsr(self):

    start(self.render_folder,self.videoName,self.input_file,1)
    os.chdir(f'{thisdir}/realesrgan-vulkan-models')
    realESRGAN(self)
def realESRGAN(self):

        self.endNum=0
        self.paused=False
        os.system(f'./realesrgan-ncnn-vulkan {self.realESRGAN_Model} -i "{self.render_folder}/{self.videoName}_temp/input_frames" -o "{self.render_folder}/{self.videoName}_temp/output_frames" {return_gpu_settings(self)} ')
        if os.path.exists(f'{self.render_folder}/{self.videoName}_temp/output_frames/') == False or os.path.isfile(f'{self.render_folder}/{self.videoName}_temp/audio.m4a') == False:
                show_on_no_output_files(self)
        else:
                if self.paused == False:
                    self.output_file = end(self.render_folder,self.videoName,self.input_file,1,self.output_folder, self.videoQuality,self.encoder)
                else:
                    pass
    
def startRealSR(self):
    if self.input_file != '':
        self.render='esrgan'
        settings = Settings()
        self.ui.ETAPreview.setText('ETA:')
        self.ui.processedPreview.setText('Files Processed:')
        self.ui.FPSPreview.setText('RES:')
        self.setDisableEnable(True)
        self.times = 1
        self.fps=VideoName.return_video_framerate(f'{self.input_file}')
        
        video = cv2.VideoCapture(self.input_file)
        self.videowidth = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.videoheight = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.aspectratio = self.videowidth / self.videoheight
        self.setDisableEnable(True)
        
        if settings.DiscordRPC == 'Enabled':
            start_discordRPC(self,'Upscaling')
        os.system(f'rm -rf "{self.render_folder}/{self.videoName}_temp/"')
        
        os.system(f'rm -rf "{self.render_folder}/{self.videoName}_temp/"')
        realESRGAN_Model = self.ui.Rife_Model.currentText()
        realESRGAN_Times = self.ui.Rife_Times.currentText()
        if realESRGAN_Model == 'Default':
            self.realESRGAN_Model = '-n realesrgan-x4plus -s 4'
        if realESRGAN_Model == 'Animation':
            self.realESRGAN_Model = f'-n realesr-animevideov3 -s {realESRGAN_Times}'
        Thread(target=lambda: renderRealsr(self)).start()
        self.runPB()