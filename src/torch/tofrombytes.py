import subprocess
import matplotlib.pyplot as plt
import numpy as np
from queue import Queue
from rife.rife import *
import cv2
import sys
#read

#def
readBuffer = Queue(maxsize=50)
writeBuffer = Queue(maxsize=50)
interpolation_factor = 10



command = ['ffmpeg', '-i', 'out.mp4', '-f', 'image2pipe','-vframes','2', '-ss', '00:00:00', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo','-s','1280x720', '-']
interpolate_process = Rife( interpolation_factor = interpolation_factor,
        interpolate_method = 'rife4.14',
        width=1280,
        height=720,
        half=True)
        
process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=1,
                
            )
frame_size = 1280 * 720 * 3
while True:
    try:
        chunk = process.stdout.read(frame_size)
        frame = np.frombuffer(chunk, dtype=np.uint8).reshape(
                            (720, 1280, 3)
                        )
                        
        readBuffer.put(frame)
    except:
        readBuffer.put(None)
        break
frame1 = readBuffer.get()
frame2 = readBuffer.get()

# Display the image using matplotlib


#proc
def proc_image(frame1,frame2):
    interpolate_process.run(frame1, frame2)
    writeBuffer.put(frame1)
    for i in range(interpolation_factor - 1):
                result = interpolate_process.make_inference(
                                (i + 1) * 1.0 / (interpolation_factor + 1)
                            )
                writeBuffer.put(result)
    writeBuffer.put(frame2)

def finish_render():
    writeBuffer.put(None)

proc_image(frame1,frame2)
                        

# save


def writes():
    command = ['ffmpeg', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo','-s','1280x720','-r','10', '-i', '-', 'out1.mp4']
    process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=sys.stdout,
                stderr=sys.stdout,
                universal_newlines=True,
                
            )
    while True:
            
            frame = writeBuffer.get()
            if frame is None:
                    
                    process.stdin.close()
                    process.wait()
                    isWritingDone = True
                    break
            frame = np.ascontiguousarray(frame)
            process.stdin.write(frame.tobytes())
        
            
            
writes()
    

# Save the image using PIL