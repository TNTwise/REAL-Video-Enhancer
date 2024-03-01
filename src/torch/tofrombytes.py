import subprocess
import matplotlib.pyplot as plt
import numpy as np
from queue import Queue
from rife.rife import *
import sys
from threading import Thread
import cv2
#read

#def
class Render:
    def __init__(self,input_file):
        self.readBuffer = Queue(maxsize=50)
        self.writeBuffer = Queue(maxsize=50)
        self.interpolation_factor = 10
        self.prevFrame = None
        cap = cv2.VideoCapture(input_file)
        self.initialFPS = cap.get(cv2.CAP_PROP_FPS)
        self.finalFPS = self.initialFPS*self.interpolation_factor
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.input_file = input_file
    def extractFramesToBytes(self):
        command = [f'ffmpeg', 
                   '-i', 
                   f'{self.input_file}', 
                   '-f',
                   'image2pipe',
                   '-pix_fmt', 
                   'rgb24', 
                   '-vcodec',
                   'rawvideo',
                   '-s',
                   f'{self.width}x{self.height}',
                   '-']
        self.interpolate_process = Rife( interpolation_factor = self.interpolation_factor,
                interpolate_method = 'rife4.14',
                width=1280,
                height=720,
                half=True)
                
        self.process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        
                    )
        self.frame_size = 1280 * 720 * 3
        
        
    def readThread(self):
        while True:
                
                chunk = self.process.stdout.read(self.frame_size)
                if len(chunk) < self.frame_size:
                    
                    self.process.stdout.close()
                    self.process.terminate()
                    self.readingDone = True
                    self.readBuffer.put(None)
                    print('done with read')
                    break
                frame = np.frombuffer(chunk, dtype=np.uint8).reshape(
                                    (720, 1280, 3)
                                )
                                
                self.readBuffer.put(frame)
            
    



#proc
    
    def proc_image(self,frame1,frame2):
        
        self.interpolate_process.run(frame1, frame2)
        self.writeBuffer.put(frame1)
        for i in range(self.interpolation_factor - 1):
                    result = self.interpolate_process.make_inference(
                                    (i + 1) * 1.0 / (self.interpolation_factor + 1)
                                )
                    self.writeBuffer.put(result)
        self.writeBuffer.put(frame2)
        
    def procThread(self):
        while True:
            frame = self.readBuffer.get()
            if frame is None:
                print('done with proc')
                self.writeBuffer.put(None)
                break # done with proc
            if self.prevFrame is None:
                self.prevFrame = frame
                continue
            self.proc_image(self.prevFrame,frame)
            
    
    
    def finish_render(self):
        self.writeBuffer.put(None)


                        

# save


    def FFmpegOut(self):
        print('saving')
        command = ['ffmpeg',
                   '-f', 
                   'rawvideo',
                   '-pix_fmt',
                   'rgb24',
                   '-vcodec', 
                   'rawvideo',
                   '-s',
                   f'{self.width}x{self.height}',
                   '-r',
                   f'{self.finalFPS}',
                   '-i',
                   '-',
                   'out1.mp4',
                   '-y']
        process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=sys.stdout,
                    stderr=sys.stdout,
                    universal_newlines=True,
                    
                )
        while True:
                try:
                    frame = self.writeBuffer.get()
                    if frame is None:
                            
                            process.stdin.close()
                            process.wait()
                            print('done with save')
                            break
                    frame = np.ascontiguousarray(frame)
                    process.stdin.buffer.write(frame.tobytes())
                except Exception as e:
                    print(e)
        
            
            
    


render = Render('temp.webm')
render.extractFramesToBytes()
readThread1 = Thread(target=render.readThread)
procThread1 = Thread(target=render.procThread)
readThread1.start()
procThread1.start()
render.FFmpegOut()