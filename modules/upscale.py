
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
from cv2 import VideoCapture, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT

import src.thisdir
thisdir = src.thisdir.thisdir()
homedir = os.path.expanduser(r"~")
def startRender(self):

    start(self,self.render_folder,self.videoName,self.input_file,1)
    
    realESRGAN(self)
def realESRGAN(self):
        settings = Settings()
        self.endNum=0
        self.paused=False
        img_type = self.settings.Image_Type.replace('.','')
        if self.AI == 'realesrgan-ncnn-vulkan':
            os.system(f'"{settings.ModelDir}/realesrgan/realesrgan-ncnn-vulkan" -i "{self.render_folder}/{self.videoName}_temp/input_frames" -o "{self.render_folder}/{self.videoName}_temp/output_frames" {self.realESRGAN_Model}{return_gpu_settings(self)} -f {img_type} ')
        if os.path.exists(f'{self.render_folder}/{self.videoName}_temp/output_frames/') == False:
                show_on_no_output_files(self)
        else:
                if self.paused == False:
                    self.output_file = end(self,self.render_folder,self.videoName,self.input_file,1,self.output_folder, self.videoQuality,self.encoder,'upscale')
                else:
                    pass
    
def start_upscale(self,AI):
    if self.input_file != '':
        
        self.ui.QueueButton.show()
        self.render='esrgan'
        self.AI = AI
        settings = Settings()
        self.setDisableEnable(True)
        
        if settings.DiscordRPC == 'Enabled':
            start_discordRPC(self,'Upscaling')
        self.ui.logsPreview.append(f'[Extracting Frames]')
            
        realESRGAN_Model = self.ui.Rife_Model.currentText()
        realESRGAN_Times = self.ui.Rife_Times.currentText()
        if AI == 'realesrgan-ncnn-vulkan':
            if realESRGAN_Model == 'Default':
                self.realESRGAN_Model = '-n realesrgan-x4plus -s 4'
            if realESRGAN_Model == 'Animation':
                self.realESRGAN_Model = f'-n realesr-animevideov3 -s {realESRGAN_Times}'
        Thread(target=lambda: startRender(self)).start()
        self.runPB()