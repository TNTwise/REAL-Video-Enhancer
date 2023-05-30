#This script creates a class that takes in params like "RealESRGAN or Rife", the model for the program,  the times of upscaling, and the path of the video, and the output path
# hz
import src.return_data as return_data
import os
import src.settings as settings
import glob
from threading import Thread
thisdir= os.getcwd()
homedir = os.path.expanduser(r"~")
def start(renderdir,videoName,videopath):
        global fps
        fps = return_data.Fps.return_video_fps(fr'{videopath}')
        os.system(f'rm -rf "{renderdir}/{videoName}_temp/"')
        
        os.mkdir(f'{renderdir}/{videoName}_temp/')
        os.mkdir(f'{renderdir}/{videoName}_temp/input_frames')
       
        os.mkdir(f'{renderdir}/{videoName}_temp/transitions')
        os.system(f'ffmpeg -i "{videopath}" "{renderdir}/{videoName}_temp/input_frames/%08d.png" -y ') # Add image extraction setting here, also add ffmpeg command here as if its compiled or not
        os.system(f'ffmpeg -i "{videopath}" -vn -c:a aac -b:a 320k "{renderdir}/{videoName}_temp/audio.m4a" -y') # do same here i think maybe
        os.mkdir(f'{renderdir}/{videoName}_temp/output_frames') # this is at end due to check in progressbar to start, bad implementation should fix later....

def end(renderdir,videoName,videopath,times,outputpath,videoQuality,encoder):
        
        
        
        if return_data.ManageFiles.isfile(f'{outputpath}/{videoName}_{fps*2}fps.mp4') == True:
                i=1
                while return_data.ManageFiles.isfile(f'{outputpath}/{videoName}_{fps*2}fps({i}).mp4') == True:
                        i+=1
                output_video_file = f'{outputpath}/{videoName}_{fps*times}fps({i}).mp4' 

        else:
               output_video_file = f'{outputpath}/{videoName}_{fps*times}fps.mp4' 
        if output_video_file == '':
                output_video_file = homedir
        os.system(f'ffmpeg -framerate {fps*times} -i "{renderdir}/{videoName}_temp/output_frames/%08d.png" -i "{renderdir}/{videoName}_temp/audio.m4a" -c:v libx{encoder} -crf {videoQuality} -c:a copy "{output_video_file}" -y') #ye we gonna have to add settings up in this bish
                
        os.system(f'rm -rf "{renderdir}/{videoName}_temp/"')
        

        
                
