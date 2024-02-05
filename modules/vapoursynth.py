import subprocess
from threading import Thread
import src.programData.thisdir
thisdir = src.programData.thisdir.thisdir()
from src.programData.return_data import *
class VapourSynth:
    def __init__(self, main):
        self.main = main
    
    def start_thread(self):
        if self.main.AI == 'vapoursynth-rife-ncnn-vulkan':
            thread = Thread(target=lambda: self.start_vs_interpolate)
        thread.start()
    
    def start_vs_interpolate(self):
        
            
        if self.main.AI == 'vapoursynth-rife-ncnn-vulkan':
            with open(f'{thisdir}/files/vsrife.txt', 'w') as f:
                f.write(f'FPS, {VideoName.return_video_framerate(f"{self.main.input_file}")}\n')
                f.write(f'Multiplier, {self.main.times}\n')
                f.write(f'Model:, {self.main.ui.Rife_Model.currentText()}\n')
            command = [f'{thisdir}/models/vapoursynth/vspipe',
                       f'{thisdir}/models'
                       ]
        
        self.main.renderAI = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)    
        stdout, stderr = self.main.renderAI.communicate()