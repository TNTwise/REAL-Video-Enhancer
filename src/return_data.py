import cv2
import os 
import re
import subprocess
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
            output = subprocess.check_output('lspci | grep "VGA"', shell=True).decode('utf-8')# Gets output of VGA and grabs GPU
            device_id = output[:7] #Grabs the device id
            output = subprocess.check_output(f'lspci -v -s {device_id} | grep "size="', shell=True).decode('utf-8') #puts the device id into the other command to give me the info
            find_size_num = (re.findall(r'[\d]*[M|G|K]',output)) # finds the numbers
            number_list = [] # List to store the numbers before sorted
            sorted_number_list = []
            for i in find_size_num: #adds the numbers to a list
                if any(char.isdigit() for char in i) == True:
                    number_list.append(i)
            for i in number_list: #strips and converts the numbers to a common gigabyte format
                
                if 'M' in i:
                    i = re.sub("[^0-9]", "", i)
                    i = int(i) * 0.001
                    sorted_number_list.append(i)
            for i in number_list:
                if 'G' in i:
                    i = re.sub("[^0-9]", "", i)
                    sorted_number_list.append(i)
            sorted_number_list.sort(reverse=True)
            return sorted_number_list[0] # Returns amount of vram
        except Exception as e:
            print(f'{e}')
            return None