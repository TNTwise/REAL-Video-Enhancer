import os
import requests
import sys
import src.thisdir
thisdir = src.thisdir.thisdir()
from PyQt5.QtWidgets import  QMessageBox
from src.settings import *
from src.return_data import *
import math
import shutil
def check_if_models_exist(thisdir):
    if os.path.exists(f'{thisdir}/models/') and os.path.exists(f'{thisdir}/models/rife/') and os.path.exists(f'{thisdir}/models/realesrgan/') and os.path.exists(f'{thisdir}/models/waifu2x/') :
        return True
    else:
        return False
    
def check_if_online():
    online=False
    try:
        requests.get('https://raw.githubusercontent.com/')
        online=True
    except:
        msg = QMessageBox()
        msg.setWindowTitle(" ")
        msg.setText(f"You are offline, please connect to the internet to download the models.")
        sys.exit(msg.exec_())
    return online

def check_if_free_space(RenderDir):
        

        return shutil.disk_usage(f'{RenderDir}').free
    
def check_if_enough_space(input_file,render,times):
    settings = Settings()
    img_type = settings.Image_Type
    if img_type == '.jpg':
        multiplier = 1/10
    if img_type == '.webp':
        multiplier= 1/30
    if img_type == '.png':
        multiplier = 2.34/3
    resolution = VideoName.return_video_resolution(input_file)
    frame_count = VideoName.return_video_frame_count(input_file)
    resolution_multiplier =  math.ceil(resolution[1] * resolution[0])
    
    # 1080p = 1, make adjustments for other resolutions
    print(f'{resolution_multiplier} {frame_count}  {multiplier}  ')
    full_extraction_size = (resolution_multiplier * frame_count * multiplier)
    free_space = check_if_free_space(settings.RenderDir)
    if settings.RenderType == 'Classic':
         #calculates the anount of storage necessary for the original extraction, in bits
        print(f'{full_extraction_size} KB')

        # add full_extraction_size to itself times the multiplier of the interpolation amount for rife
        if render == 'esrgan':
            return True,0,0
        if render == 'rife':
            if img_type == '.png':
                full_size = full_extraction_size + full_extraction_size * times
                
                return full_size < free_space, full_size / (1024 ** 3), free_space/ (1024 ** 3)
            if img_type == '.jpg':
                
                full_size = full_extraction_size + full_extraction_size * times * 5
                return full_size < free_space, full_size/ (1024 ** 3), free_space/ (1024 ** 3)
            if img_type == '.webp':
                
                full_size = full_extraction_size + full_extraction_size * times * 4
                return full_size < free_space, full_size/ (1024 ** 3), free_space/ (1024 ** 3)
    else:
        return full_extraction_size*2 < free_space, full_extraction_size*2/ (1024 ** 3), free_space/ (1024 ** 3)
            
def check_for_individual_models():
    return_list = []
    if os.path.exists(f'{thisdir}/models/'):
        if os.path.exists(f'{thisdir}/models/rife/'):
            return_list.append('Rife')
        if os.path.exists(f'{thisdir}/models/realesrgan/'):
            return_list.append('RealESRGAN')
        if os.path.exists(f'{thisdir}/models/waifu2x/'):
            return_list.append('Waifu2X')
        if os.path.exists(f'{thisdir}/models/realcugan/'):
            return_list.append('RealCUGAN')
        if os.path.exists(f'{thisdir}/models/cain/'):
            return_list.append('Cain')
        if len(return_list) >0:
            return return_list
        
    return None