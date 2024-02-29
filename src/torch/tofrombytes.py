import subprocess
import matplotlib.pyplot as plt
import numpy as np
from queue import Queue
from rife.rife import *
import cv2

readBuffer = Queue()
command = ['ffmpeg', '-i', 'out.mp4', '-f', 'image2pipe','-vframes','2', '-ss', '00:00:00', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo','-s','1280x720', '-']
interpolate_process = Rife( interpolation_factor = 2,
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
for i in range(2):
    
    chunk = process.stdout.read(frame_size)
    frame = np.frombuffer(chunk, dtype=np.uint8).reshape(
                        (720, 1280, 3)
                    )
                    
    readBuffer.put(frame)

frame1 = readBuffer.get()
frame2 = readBuffer.get()

# Display the image using matplotlib



interpolate_process.run(frame1, frame2)
for i in range(10 - 1):
            result = interpolate_process.make_inference(
                            (i + 1) * 1.0 / (2 + 1)
                        )
            plt.imsave(f'{i}.png', result)

                        
from PIL import Image
from io import BytesIO

plt.imsave('frame1.png', frame1)

plt.imsave('frame2.png', frame2)
# Save the image using PIL