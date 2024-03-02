import subprocess
import numpy as np
from queue import Queue

from .rife.rife import *
from src.programData.thisdir import thisdir
thisdir = thisdir()
import sys
from threading import Thread
import cv2
import matplotlib.pyplot as plt
import io
#read
# Calculate eta by time remaining divided by speed
# add scenedetect by if frame_num in transitions in proc_frames
#def
class Render:
    def __init__(self,main,input_file,output_file,times):
        
        self.main = main
        
        self.readBuffer = Queue(maxsize=50)
        self.writeBuffer = Queue(maxsize=50)
        times = 2
        self.interpolation_factor = times
        self.prevFrame = None
        cap = cv2.VideoCapture(input_file)
        self.initialFPS = cap.get(cv2.CAP_PROP_FPS)
        self.finalFPS = self.initialFPS*self.interpolation_factor
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.input_file = input_file
        self.output_file = output_file
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
                width=self.width,
                height=self.height,
                half=True)
                
        self.process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        
                    )
        self.frame_size = self.width * self.height * 3
        
        
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
                                    (self.height, self.width, 3)
                                )
                                
                self.readBuffer.put(frame)
            
    



#proc
    
    def proc_image(self,frame1,frame2):
        
        self.interpolate_process.run(frame1, frame2)
        self.writeBuffer.put(frame1)
        for i in range(self.interpolation_factor - 1):
                    
                    result = self.interpolate_process.make_inference(
                                    (i+1) * 1. / (self.interpolation_factor)
                                )
                    self.main.imageDisplay=result
                    self.writeBuffer.put(result)
        
        
    def procThread(self):
        i=0
        while True:
            frame = self.readBuffer.get()
            
            if frame is None:
                print('done with proc')
                self.writeBuffer.put(self.prevFrame)
                self.writeBuffer.put(None)
                break # done with proc

            if self.prevFrame is None:
                self.prevFrame = frame
                continue
            self.main.imageDisplay=frame
            plt.imsave(f'{i}.png',self.main.imageDisplay)
            

            self.proc_image(self.prevFrame,frame)
            self.prevFrame = frame
            i+=1
    
    def finish_render(self):
        self.writeBuffer.put(None)

    def returnLatestFrame(self):
        try:
            return self.prevFrame
        except:
            print('No frame to return!')
        
    
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
                   f'{self.output_file}',
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
                            self.main.output_file = self.output_file
                            self.main.CudaRenderFinished = True
                            break
                    frame = np.ascontiguousarray(frame)
                    process.stdin.buffer.write(frame.tobytes())
                except Exception as e:
                    print(e)
        
            
            
def startRender(self,inputFile,outputFile,times):
    render = Render(self,inputFile,outputFile,int(times))
    render.extractFramesToBytes()
    readThread1 = Thread(target=render.readThread)
    procThread1 = Thread(target=render.procThread)
    renderThread1 = Thread(target=render.FFmpegOut)
    readThread1.start()
    procThread1.start()
    renderThread1.start()
    
