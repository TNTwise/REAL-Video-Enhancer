#This script creates a class that takes in params like "RealESRGAN or Rife", the model for the program,  the times of upscaling, and the path of the video, and the output path
# hz
import src.return_data as return_data
import os
import src.settings as settings
import glob
thisdir= os.getcwd()
homedir = os.path.expanduser(r"~")


def start(renderdir,videoName,videopath):
        os.system(f'rm -rf "{renderdir}/{videoName}/"')
        
        os.mkdir(f'{renderdir}/{videoName}/')
        os.mkdir(f'{renderdir}/{videoName}/input_frames')
       
        os.mkdir(f'{renderdir}/{videoName}/transitions')
        os.system(f'ffmpeg -i "{videopath}" "{renderdir}/{videoName}/input_frames/%08d.png" ') # Add image extraction setting here, also add ffmpeg command here as if its compiled or not
        os.system(f'ffmpeg -i "{videopath}" -vn -c:a aac -b:a 320k "{renderdir}/{videoName}/audio.m4a" -y') # do same here i think maybe
        os.mkdir(f'{renderdir}/{videoName}/output_frames')
def end(renderdir,videoName,videopath,times,outputpath):
        
        fps = return_data.Fps.return_video_fps(fr'{videopath}')
        
        if return_data.ManageFiles.isfile(f'{outputpath}/{videoName}_{fps*2}fps.mp4') == True:
                i=1 
                while return_data.ManageFiles.isfile(f'{outputpath}/{videoName}_{fps*2}fps({i}).mp4') == True:
                        i+=1
                output_video_file = f'{outputpath}/{videoName}_{fps*2}fps({i}).mp4' 

        else:
               output_video_file = f'"{outputpath}/{videoName}_{fps*2}fps.mp4"' 
        os.system(f'ffmpeg -framerate {fps*times} -i "{renderdir}/{videoName}/output_frames/%08d.png" -crf 18 -c:a copy "{output_video_file}"') #ye we gonna have to add settings up in this bish
                
        os.system(f'rm -rf "{renderdir}/{videoName}/"')
        
def start_rife(model,times,videopath,outputpath,renderdir=thisdir):
        
        videoName = return_data.VideoName.return_video_name(fr'{videopath}')
        start(renderdir,videoName,videopath)

        os.system(f'"{thisdir}/rife-vulkan-models/rife-ncnn-vulkan" -m  {model} -i {renderdir}/{videoName}/input_frames/ -o {renderdir}/{videoName}/output_frames/')

        end(renderdir,videoName,videopath,times,outputpath)
        
                
