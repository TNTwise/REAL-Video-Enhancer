import subprocess
import matplotlib.pyplot as plt
import numpy as np
from queue import Queue
from rife.rife import *
import cv2

interpolation_factor = 2
interpolate_process = Rife( interpolation_factor,
        interpolate_method = 'rife4.15',
        width=960,
        height=576,
        half=True)


frame_size = 960 * 576 * 3


frame1 = cv2.imread("in0.png")
#frame1 = cv2.imread("outputcuda/0.png")

frame2 = cv2.imread("in1.png")

# Display the image using matplotlib

# get 
factor_num = 5
factor_den = 2
interpolate_process.run(frame1, frame2)
for i in range(factor_num - factor_den):
            result = interpolate_process.make_inference(
                ((i/factor_num) + 1) * 1.0 / (interpolation_factor)
            )
            plt.imsave(f'outputcuda/{i}.png', result)


from PIL import Image
from io import BytesIO

plt.imsave('frame1.png', frame1)

plt.imsave('frame2.png', frame2)
# Save the image using PIL 