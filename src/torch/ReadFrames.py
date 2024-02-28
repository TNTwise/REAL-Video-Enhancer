import subprocess
import numpy as np
from queue import Queue
from RenderFrames import *
class ReadBuffer:
    def __init__(self,video_path):
        self.video_path = video_path
        self.queueSize = 50
        
        #can be built later
        self.command = ['ffmpeg', '-i', self.video_path, '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-']
    def start(self, queue: Queue = None, verbose: bool = False):
        
        
        self.readBuffer = queue if queue is not None else Queue(maxsize=self.queueSize)
        

       

        try:
            process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=1,
                
            )

            self.readingDone = False
            frame_size = 1280 * 720 * 3
            self.decodedFrames = 0

            while True:
                
                    
                    chunk = process.stdout.read(frame_size)
                    

                    frame = np.frombuffer(chunk, dtype=np.uint8).reshape(
                        (720, 1280, 3)
                    )

                    self.readBuffer.put(frame)
                    self.decodedFrames += 1
                

        except Exception as e:
            
            self.readingDone = True
            self.readBuffer.put(None)
    def render(self):
        frame = self.readBuffer.get() if not self.readBuffer.empty() else None
        procInterp = Interp()
       
        while True:
           procInterp.processFrame(frame)
           
readbuffer = ReadBuffer('test.webm')
readbuffer.start()
readbuffer.render()