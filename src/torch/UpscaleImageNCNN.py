import numpy as np
from upscale_ncnn_py import UPSCALE
import cv2


class UpscaleNCNN:
    def __init__(self, model, num_threads, scale):
        self.model = UPSCALE(
            gpuid=0, model_str=model, num_threads=num_threads, scale=scale
        )

    def UpscaleImage(self, image):
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
