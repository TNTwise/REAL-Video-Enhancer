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
    
def check_if_online(dont_check=False):
    online=False
    try:
        requests.get('https://raw.githubusercontent.com/')
        online=True
    except:
        if dont_check ==False:
            msg = QMessageBox()
            msg.setWindowTitle(" ")
            msg.setText(f"You are offline, please connect to the internet to download the models.")
            sys.exit(msg.exec_())
        pass
    return online

def check_if_free_space(RenderDir):
        

        return shutil.disk_usage(f'{RenderDir}').free
def check_if_enough_space_for_install(size_in_bytes):
    free_space= shutil.disk_usage(f'{thisdir}').free
    return free_space >  size_in_bytes

def check_if_flatpak():
    if 'FLATPAK_ID' in os.environ:
        return True
    return False


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
            rnd = round(resolution*.001)
            if rnd < 1:
                rnd = 1
            if img_type == '.png':
                full_size = full_extraction_size + full_extraction_size * times * rnd
                
            if img_type == '.jpg':
                
                full_size = full_extraction_size + full_extraction_size * times * 5*rnd
            if img_type == '.webp':
                
                full_size = full_extraction_size + full_extraction_size * times * 4*rnd
            return full_size < free_space, full_size/ (1024 ** 3), free_space/ (1024 ** 3)
        
        if render == 'rife':
            if img_type == '.png':
                full_size = full_extraction_size + full_extraction_size * times
                
            if img_type == '.jpg':
                
                full_size = full_extraction_size + full_extraction_size * times * 5
            if img_type == '.webp':
                
                full_size = full_extraction_size + full_extraction_size * times * 4
            return full_size < free_space, full_size/ (1024 ** 3), free_space/ (1024 ** 3)
    else:
        return full_extraction_size*5 < free_space, full_extraction_size*2/ (1024 ** 3), free_space/ (1024 ** 3)
            
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
        if os.path.exists(f'{thisdir}/models/ifrnet/'):
            return_list.append('IFRNET')
        if len(return_list) >0:
            return return_list
        
    return None

def check_for_write_permissions(dir):


         if 'FLATPAK_ID' in os.environ or "container" in os.environ:
            import subprocess

            command = 'flatpak info --show-permissions io.github.tntwise.REAL-Video-Enhancer'

            result = subprocess.run(command, shell=True, capture_output=True, text=True)

            output = result.stdout.split('\n')
            output_2=[]
            for i in output:
                if len(i) > 0 and i != '\n':
                    output_2.append(i)
            
            directories_with_permissions=[]
            for i in output_2:
                if 'filesystems=' in i:
                    i=i.split(';')
                    print(i)
                    s=[]
                    for e in  i:
                        if len(e) > 0 and i != '\n':
                            s.append(e)
                    for j in s:
                        j=j.replace('filesystems=','')
                        j=j.replace('xdg-','/home/pax/')
                        j+='/'
                        directories_with_permissions.append(j)
                    break
            for i in directories_with_permissions:
                if dir.lower() in i.lower() or 'io.github.tntwise.real-video-enhancer' in dir.lower():
                    
                    return True
                
            return False
         else:
            if os.access(dir, os.R_OK) and os.access(dir, os.W_OK):
                print('has access')
                return True
            return False

def check_for_each_binary():
    if os.path.isfile(f'{thisdir}/bin/ffmpeg') and os.path.isfile(f'{thisdir}/bin/glxinfo') and os.path.isfile(f'{thisdir}/bin/glxinfo') and os.path.isfile(f'{thisdir}/bin/glxinfo'):
        return True
    return False

def check_for_updates():
    if check_if_online() == True:
        settings = Settings()
        if settings.UpdateChannel == 'Beta':
            latest='https://github.com/TNTwise/REAL-Video-Enhancer-BETA/releases/latest'
        if settings.UpdateChannel == 'Stable':
            latest='https://github.com/TNTwise/REAL-Video-Enhancer/releases/latest'
        latest = requests.get(latest) 
        latest = latest.url
        latest = re.findall(r'[\d]*$', latest)
        latest = latest[0]
        if int(latest) >= int(settings.Version):
            return
        else:
            new_ver = requests.get(f'https://github.com/TNTwise/REAL-Video-Enhancer-BETA/releases/download/{latest}/REAL-Video-Enhancer-x86_64.AppImage')