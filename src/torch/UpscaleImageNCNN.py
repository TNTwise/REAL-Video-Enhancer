import numpy as np
from upscale_ncnn_py import UPSCALE
from realcugan_ncnn_py import Realcugan
import cv2


class UpscaleNCNN:
    def __init__(self, model, num_threads, scale, gpuid=0):
        self.model = UPSCALE(
            gpuid=gpuid, model_str=model, num_threads=num_threads, scale=scale
        )

    def UpscaleImage(self, image):
        return self.model.process_cv2(image)

class UpscaleCuganNCNN:
    def __init__(self,model="models-se",models_path="",num_threads=2,scale=2, gpuid=0,noise=0):
        self.model = Realcugan(gpuid=gpuid,
                               models_path=models_path,
                               model=model,
                               scale=scale,
                               num_threads=num_threads,
                               noise=noise)

    def UpscaleImage(self,image):
        return self.model.process_cv2(image) 
if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt
    from realesrgan_ncnn_py import Realesrgan

    realesrgan = Realesrgan(gpuid=0)
    image = cv2.imdecode(np.fromfile("in0.png", dtype=np.uint8), cv2.IMREAD_COLOR)
    image = realesrgan.process_cv2(image)
    cv2.imencode(".jpg", image)[1].tofile("output_cv2.jpg")

    upscale = UpscaleNCNN()
    image = cv2.imread("in0.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert the image to a NumPy array
    image_array = np.array(image)

    plt.imsave("in.png", image_array)
    upscaled_image = upscale.UpscaleImage(image_array)
    plt.imsave("out.png", upscaled_image)
