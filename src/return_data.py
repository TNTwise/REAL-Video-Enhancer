import cv2
import os 
import re
import subprocess
import shutil
import src.thisdir

thisdir = src.thisdir.thisdir()
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


def get_vram_amount():
    try:
        # Run the nvidia-smi command and capture the output
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])

        # Convert the output to a string and strip any leading/trailing whitespace
        vram_str = output.decode("utf-8").strip()

        # Convert the VRAM amount to an integer
        vram_amount = int(vram_str)

        return int(vram_amount // 1000)
    except Exception as e:
        # Handle any exceptions that may occur (e.g., nvidia-smi not found, error running the command)
        print(f"Error: {e}")
        return get_integrated_vram()

# Get the VRAM amount and store it in a variable



def get_dedicated_vram():
    try:
        command = f"./{thisdir}/bin/glxinfo | grep 'Dedicated video memory'"
        vram_available = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, check=True)
        vram = vram_available.stdout.split(":")[1].replace('MB', '').strip()
        return int(vram) // 1000
    except subprocess.CalledProcessError:
        return get_vram_amount()
        
        

def get_integrated_vram():

    return 1
def ceildiv(a, b):
    return -(a // -b)

def returnCodec(codec):
    if codec=='264':
        return 'libx264'
    if codec=='265':
        return 'libx265'
    if codec == 'VP9':
        return 'libvpx-vp9'
    if codec == 'ProRes':
        return 'prores -profile:v 2'
    if codec == 'Lossless':
        return 'copy'
    if codec == 'AV1':
        return 'libaom-av1'
'''
Codec Options



libvpx-vp9 - address bitrate based on quality setting

prores: 
pair these to quality settings
-profile:
    -1: auto (default)
    0: proxy ≈ 45Mbps YUV 4:2:2
    1: lt ≈ 102Mbps YUV 4:2:2
    2: standard ≈ 147Mbps YUV 4:2:2
    3: hq ≈ 220Mbps YUV 4:2:2
    4: 4444≈ 330Mbps YUVA 4:4:4:4
    5: 4444xq ≈ 500Mbps YUVA 4:4:4:4




'''

def returnContainer(codec):
    print(codec)
    if '264' in codec:
        return 'mp4'
    if '265' in codec:
        return 'mp4'
    if 'vp9' in codec.lower():
        return 'mp4'
    if 'prores' in codec.lower():
        return 'mov'
    if codec == 'copy' or codec == 'Lossless':
        return 'mkv'
    if 'av1' in codec.lower():
        return 'mkv'
    
    
    
class HardwareInfo:
    
    def get_video_memory_linux():
        
        card = 0
        while card < 10:
            if os.path.exists(f'/sys/class/drm/card{card}/device/mem_info_vram_total'):
                return read_vram(card)
            else:
                card+=1
                continue
        
        vram = get_dedicated_vram()
        
        return vram



