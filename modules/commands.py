
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
    if int(self.gpuMemory) < 1:
        gpu_usage = f'-j 1:1:1'
    else:
        num = int(int(self.gpuMemory))
        gpu_usage = f'-j {num}:{num}:{num}'
    return gpu_usage

def start(self,renderdir,videoName,videopath,times):
        
        if self.localFile == True:
                self.fps = return_data.Fps.return_video_fps(fr'{videopath}')
        
                global height
                global width
                width,height = return_data.VideoName.return_video_resolution(videopath)
                '''with open(f'{renderdir}/{videoName}_temp/data.txt', 'w') as f:
                f.write(f'{times}')'''
        return_data.ManageFiles.create_folder(f'{renderdir}/{videoName}_temp/')
        return_data.ManageFiles.create_folder(f'{renderdir}/{videoName}_temp/input_frames')
       
        if self.localFile == True:
                if self.settings.Image_Type != '.webp':
                        os.system(f'{thisdir}/bin/ffmpeg -i "{videopath}" -q:v 1 "{renderdir}/{videoName}_temp/input_frames/%08d{self.settings.Image_Type}" -y ') 
                else:
                        os.system(f'{thisdir}/bin/ffmpeg -i "{videopath}" -c:v libwebp -q:v 100 "{renderdir}/{videoName}_temp/input_frames/%08d.webp" -y ') 
                os.system(f'{thisdir}/bin/ffmpeg -i "{videopath}" -vn -c:a aac -b:a 320k "{renderdir}/{videoName}_temp/audio.m4a" -y') # do same here i think maybe
        else:
               command = self.extract_frames_from_youtube_video_command.replace('out.mp4',f'-q:v 1 "{renderdir}/{videoName}_temp/input_frames/%08d{self.settings.Image_Type}" -y')
               command = command.replace('output.m4a',f'"{renderdir}/{videoName}_temp/audio.m4a"')
               os.system(command)
        return_data.ManageFiles.create_folder(f'{renderdir}/{videoName}_temp/output_frames') # this is at end due to check in progressbar to start, bad implementation should fix later....


def end(self,renderdir,videoName,videopath,times,outputpath,videoQuality,encoder,mode='interpolation'):
        
        
        if outputpath == '':
                outputpath = homedir
        if mode == 'interpolation':
                if return_data.ManageFiles.isfile(f'{outputpath}/{videoName}_{int(self.fps*times)}fps.mp4') == True:
                        i=1
                        while return_data.ManageFiles.isfile(f'{outputpath}/{videoName}_{int(self.fps*times)}fps({i}).mp4') == True:
                                i+=1
                        output_video_file = f'{outputpath}/{videoName}_{int(self.fps*times)}fps({i}).mp4' 

                else:
                        output_video_file = f'{outputpath}/{videoName}_{int(self.fps*times)}fps.mp4' 
        if mode == 'upscale': # add upscale/realesrgan resolution bump here
                upscaled_res = f'{int(width*self.resIncrease)}x{int(height*self.resIncrease)}'
                if return_data.ManageFiles.isfile(f'{outputpath}/{videoName}_{upscaled_res}.mp4') == True:
                        i=1
                        while return_data.ManageFiles.isfile(f'{outputpath}/{videoName}_{upscaled_res}({i}).mp4') == True:
                                i+=1
                        output_video_file = f'{outputpath}/{videoName}_{upscaled_res}({i}).mp4' 

                else:
                        output_video_file = f'{outputpath}/{videoName}_{upscaled_res}.mp4'
        os.system(f'./bin/ffmpeg -framerate {self.fps*times} -i "{renderdir}/{videoName}_temp/output_frames/%08d{self.settings.Image_Type}" -i "{renderdir}/{videoName}_temp/audio.m4a" -c:v libx{encoder} -crf {videoQuality} -c:a copy  -pix_fmt yuv420p "{output_video_file}" -y') #ye we gonna have to add settings up in this bish
        os.system(f'rm -rf "{renderdir}/{videoName}_temp/audio.m4a"')
        
        os.system(f'rm -rf "{renderdir}/{videoName}_temp/"')
        os.chdir(thisdir)
        return output_video_file