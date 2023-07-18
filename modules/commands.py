
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


def return_gpu_settings(self):
    if self.gpuMemory <= 2.0:
        gpu_usage = ''
    else:
        num = int(int(self.gpuMemory)+1)
        gpu_usage = f'-j {num}:{num}:{num}'
    return gpu_usage

def start(renderdir,videoName,videopath,times):
        global fps
        fps = return_data.Fps.return_video_fps(fr'{videopath}')
        
        global height
        global width
        width,height = return_data.VideoName.return_video_resolution(videopath)
        '''with open(f'{renderdir}/{videoName}_temp/data.txt', 'w') as f:
            f.write(f'{times}')'''
        return_data.ManageFiles.create_folder(f'{renderdir}/{videoName}_temp/')
        return_data.ManageFiles.create_folder(f'{renderdir}/{videoName}_temp/input_frames')
       
        
        
        os.system(f'./bin/ffmpeg -i "{videopath}" "{renderdir}/{videoName}_temp/input_frames/%08d.png" -y ') # Add image extraction setting here, also add ffmpeg command here as if its compiled or not
        os.system(f'./bin/ffmpeg -i "{videopath}" -vn -c:a aac -b:a 320k "{renderdir}/{videoName}_temp/audio.m4a" -y') # do same here i think maybe
        return_data.ManageFiles.create_folder(f'{renderdir}/{videoName}_temp/output_frames') # this is at end due to check in progressbar to start, bad implementation should fix later....


def end(self,renderdir,videoName,videopath,times,outputpath,videoQuality,encoder,mode='interpolation'):
        settings = Settings()
        
        
        if outputpath == '':
                outputpath = homedir
        if mode == 'interpolation':
                if return_data.ManageFiles.isfile(f'{outputpath}/{videoName}_{int(fps*times)}fps.mp4') == True:
                        i=1
                        while return_data.ManageFiles.isfile(f'{outputpath}/{videoName}_{int(fps*times)}fps({i}).mp4') == True:
                                i+=1
                        output_video_file = f'{outputpath}/{videoName}_{int(fps*times)}fps({i}).mp4' 

                else:
                        output_video_file = f'{outputpath}/{videoName}_{int(fps*times)}fps.mp4' 
        if mode == 'upscale': # add upscale/realesrgan resolution bump here
                upscaled_res = f'{int(width*self.resIncrease)}x{int(height*self.resIncrease)}'
                if return_data.ManageFiles.isfile(f'{outputpath}/{videoName}_{upscaled_res}.mp4') == True:
                        i=1
                        while return_data.ManageFiles.isfile(f'{outputpath}/{videoName}_{upscaled_res}({i}).mp4') == True:
                                i+=1
                        output_video_file = f'{outputpath}/{videoName}_{upscaled_res}({i}).mp4' 

                else:
                        output_video_file = f'{outputpath}/{videoName}_{upscaled_res}.mp4'
        os.system(f'./bin/ffmpeg -framerate {fps*times} -i "{renderdir}/{videoName}_temp/output_frames/%08d{settings.Image_Type}" -i "{renderdir}/{videoName}_temp/audio.m4a" -c:v libx{encoder} -crf {videoQuality} -c:a copy  -pix_fmt yuv420p "{output_video_file}" -y') #ye we gonna have to add settings up in this bish
        os.system(f'rm -rf "{renderdir}/{videoName}_temp/audio.m4a"')
        
        os.system(f'rm -rf "{renderdir}/{videoName}_temp/"')
        os.chdir(thisdir)
        return output_video_file