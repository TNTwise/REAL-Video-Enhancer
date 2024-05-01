from upscale_ncnn_py import UPSCALE
from realcugan_ncnn_py import Realcugan


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
