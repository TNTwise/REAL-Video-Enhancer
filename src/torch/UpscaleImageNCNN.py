from upscale_ncnn_py import UPSCALE
from realcugan_ncnn_py import Realcugan
import numpy as np

# import ncnn
import cv2


class UpscaleNCNN:
    def __init__(
        self, model: str, num_threads, scale, gpuid=0, width=1920, height=1080
    ):
        self.model = UPSCALE(
            gpuid=gpuid, model_str=model, num_threads=num_threads, scale=scale
        )
        self.width = width
        self.height = height
        self.scale = scale
        """model = model + '.param'
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = True
        self.net.load_param(model.replace('.bin','.param'))
        self.net.load_model(model.replace('.param','.bin'))
        """

    """def NCNNImageMatFromNP(self, npArray: np.array) -> ncnn.Mat:
        return ncnn.Mat.from_pixels(
            npArray,
            ncnn.Mat.PixelType.PIXEL_BGR,
            self.width,
            self.height,
        )

    def NormalizeImage(self, mat, norm_vals):
        mean_vals = []
        mat.substract_mean_normalize(mean_vals, norm_vals)

    def ClampNPArray(self, nparray: np.array) -> np.array:
        min_val = np.min(nparray)
        max_val = np.max(nparray)
        if min_val < 0 or max_val > 255:
            nparray = ((nparray - min_val) / (max_val - min_val)) * 255
        return nparray

    def ProcessNCNN(self, frame: np.array) -> np.asarray:
        ex = self.net.create_extractor()
        frame = self.NCNNImageMatFromNP(frame)
        # norm
        self.NormalizeImage(mat=frame, norm_vals=[1 / 255.0, 1 / 255.0, 1 / 255.0])
        # render frame
        ex.input("data", frame)
        ret, frame = ex.extract("output")

        # norm
        self.NormalizeImage(mat=frame, norm_vals=[255.0, 255.0, 255.0])

        frame = np.ascontiguousarray(frame)
        frame = self.ClampNPArray(frame)
        frame = frame.transpose(1, 2, 0)
        return np.ascontiguousarray(frame, dtype=np.uint8)"""

    def UpscaleImage(self, image):
        return np.ascontiguousarray(
            np.frombuffer(
                self.model.process_bytes(image, self.width, self.height, 3),
                dtype=np.uint8,
            ).reshape(self.height * self.scale, self.width * self.scale, 3)
        )


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
        self.scale = scale
        self.model = Realcugan(
            gpuid=gpuid,
            models_path=models_path,
            model=model,
            scale=scale,
            num_threads=num_threads,
            noise=noise,
        )

    def UpscaleImage(self, image):
        return np.ascontiguousarray(
            np.frombuffer(
                self.model.process_bytes(image, self.width, self.height, 3),
                dtype=np.uint8,
            ).reshape(self.height * self.scale, self.width * self.scale, 3)
        )
