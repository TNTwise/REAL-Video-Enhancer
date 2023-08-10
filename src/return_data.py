import cv2
import os 
import re
import subprocess
import shutil
from pyopencl import *
class Fps:
    def return_video_fps(videopath):
        video=cv2.VideoCapture(fr'{videopath}')
        return video.get(cv2.CAP_PROP_FPS)
    
class VideoName:
    def return_video_name(videopath):
        return os.path.basename(videopath)
    def return_video_framerate(videopath):
        video = cv2.VideoCapture(videopath)
        return video.get(cv2.CAP_PROP_FPS)
    def return_video_frame_count(videopath):
        video = cv2.VideoCapture(videopath)
        return video.get(cv2.CAP_PROP_FRAME_COUNT)
    def return_video_resolution(videopath):
        video = cv2.VideoCapture(videopath)
        width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return [width,height]
class ManageFiles:
    
    def create_folder(folderpath):
        if os.path.exists(folderpath) == False:
            os.mkdir(folderpath)

    def create_file(filepath):
        if os.path.isfile(filepath) == False:
            os.mknod(filepath)

    def isfile(filepath):
        return os.path.isfile(filepath)
            
    def isfolder(folderpath):
        return os.path.exists(folderpath)
def read_vram(card):
        with open(f'/sys/class/drm/card{card}/device/mem_info_vram_total', 'r') as f:
                    for line in f:
                        line = line.replace('\n','')
                        line = int(int(line)/1000000000)
                        return line
class HardwareInfo:
    
    def get_video_memory_linux():
        
        card = 0
        while card < 10:
            if os.path.exists(f'/sys/class/drm/card{card}/device/mem_info_vram_total'):
                return read_vram(card)
            else:
                card+=1
                continue
        

       
        try:
            import subprocess
            command = "glxinfo | grep 'Dedicated video memory'"
            vram_available = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, check=True)
            vram = vram_available.stdout.split(":")[1].replace('MB','')

            return int(vram / 1000)
        except Exception as e:
            print(f'{e}')
            return None
    