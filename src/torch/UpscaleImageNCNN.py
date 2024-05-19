from upscale_ncnn_py import UPSCALE
from realcugan_ncnn_py import Realcugan
import numpy as np

class UpscaleNCNN:
    def __init__(self, model, num_threads, scale, gpuid=0,width=1920,height=1080):
        self.model = UPSCALE(
            gpuid=gpuid, model_str=model, num_threads=num_threads, scale=scale
        )
        self.width = width
        self.height = height
    def UpscaleImage(self, image):
        
        
        image = np.ascontiguousarray(

                np.frombuffer(image,dtype=np.uint8).reshape(self.height,self.width,3)
            )
        
        return self.model.process_cv2(image)


class UpscaleCuganNCNN:
    def __init__(
        self,
        model="models-se",
        models_path="",
        num_threads=2,
        scale=2,
        gpuid=0,
        noise=0,
        width: int = 1920,
        height: int = 1080,
    ):
        self.width = width
        self.height = height
        self.model = Realcugan(
            gpuid=gpuid,
            models_path=models_path,
            model=model,
            scale=scale,
            num_threads=num_threads,
            noise=noise,
        )
        
    def UpscaleImage(self, image):
        image = np.ascontiguousarray(
                np.frombuffer(image,dtype=np.uint8).reshape(self.height,self.width,3)
            )
        return np.ascontiguousarray(self.model.process_bytes(image))
