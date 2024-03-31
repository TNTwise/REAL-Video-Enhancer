from spandrel import (
    ImageModelDescriptor,
    ModelDescriptor,
    ModelLoader,
    MaskedImageModelDescriptor,
)

try:
    from src.programData.thisdir import thisdir

    thisdir = thisdir()
except:
    thisdir = "/home/pax/.local/share/REAL-Video-Enhancer/"
import torch as torch
from torch.nn import functional as F
import cv2
import numpy as np
# from upscale_ncnn_py import UPSCALE

# from realcugan_ncnn_py import Realcugan


class UpscaleCUDA:
    def __init__(self, width, height, model):
        self.width = width
        self.height = height

        self.model = ModelLoader().load_from_file(model)
        assert isinstance(self.model, ImageModelDescriptor)
        self.model.cuda().eval()  # gonna have to put cuda back in here lmfaooooooo
        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_available else "cpu")

    @torch.inference_mode()
    def UpscaleImage(self, npArr):
        image = (
            torch.from_numpy(npArr)
            .to(self.device, non_blocking=True)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        image = image.contiguous(memory_format=torch.channels_last)

        output = self.model(image)
        # output = output[:, :, : self.width, : self.height]
        output = output.squeeze(0).permute(1, 2, 0).mul_(255).clamp_(0, 255).byte()

        return output.cpu().numpy()


"""class UpscaleNCNN:
    def __init__(self):
        #self.model = UPSCALE(gpuid=0, model=12,num_threads=2)
        pass
        self.model = UPSCALE(gpuid=0,model=7,num_threads=4)
    def UpscaleImage(self,image):
        image = self.model.process_cv2(image)
        return image

if __name__ == '__main__':
    from PIL import Image
    import matplotlib.pyplot as plt
    from realesrgan_ncnn_py import Realesrgan
    realesrgan = Realesrgan(gpuid=0)
    image = cv2.imdecode(np.fromfile("in0.png", dtype=np.uint8), cv2.IMREAD_COLOR)
    image = realesrgan.process_cv2(image)
    cv2.imencode(".jpg", image)[1].tofile("output_cv2.jpg")

    upscale = UpscaleNCNN()
    image = cv2.imread('in0.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert the image to a NumPy array
    image_array = np.array(image)
    
    plt.imsave('in.png',image_array)
    upscaled_image = upscale.UpscaleImage(image_array)
    plt.imsave('out.png',upscaled_image)"""
