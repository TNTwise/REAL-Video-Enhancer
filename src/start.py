#This script creates a class that takes in params like "RealESRGAN or Rife", the model for the program,  the times of upscaling, and the path of the video, and the output path
import return_fps
import os
import settings
homedir = os.path.expanduser(r"~")


def start_rife(model,times,videopath,outputpath=homedir):
        fps = return_fps.Fps.return_video_fps(fr'{videopath}')
        