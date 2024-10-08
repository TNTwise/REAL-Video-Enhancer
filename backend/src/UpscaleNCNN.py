import numpy as np
import os
from time import sleep

import numpy as np
try:
    from upscale_ncnn_py import UPSCALE
    method = "upscale_ncnn_py"
    s
except:
    import ncnn
    method = "ncnn_vulkan"   
class NCNNParam:
    """
    Puts the last time an op shows up in a param in a dict
    get pixelshufflescale gets the latest pixelshuffle upsample factor in a param file
    """

    def __init__(
        self,
        paramPath,
    ):
        paramDict = {}
        with open(paramPath, "r") as f:
            for line in f.readlines():
                try:
                    paramDict[line.split()[0]] = line.split()[1::]
                except IndexError:
                    pass
        self.paramDict = paramDict

    def getPixelShuffleScale(self) -> int:
        scale = None

        for value in self.paramDict["PixelShuffle"]:
            if "0=" in value:
                scale = int(value[2])
                break
        return scale

    def getInterpScale(self) -> int:
        scale = 1
        for value in self.paramDict["Interp"]:
            if "0=" in value:
                scale = int(value[2])
                break

        return scale


def getNCNNScale(modelPath: str = "") -> int:
    paramName = os.path.basename(modelPath).lower()
    for i in range(100):
        if f"{i}x" in paramName or f"x{i}" in paramName:
            return i

    # fallback
    ncnnp = NCNNParam(modelPath + ".param")
    scale = ncnnp.getPixelShuffleScale()
    return scale



class UpscaleNCNN:
    def __init__(
        self,
        modelPath: os.PathLike,
        num_threads: int,
        scale: int,
        gpuid: int = 0,
        width: int = 1920,
        height: int = 1080,
        tilesize: int = 0,
    ):
        # only import if necessary

        self.gpuid = gpuid
        self.modelPath = modelPath
        self.scale = scale
        self.tilesize = tilesize
        self.width = width
        self.height = height
        self.scale = scale
        self.threads = num_threads
        self.mean_vals = []
        self.norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
        self._load()

    def _load(self):
        if method == "ncnn_vulkan":
            self.net = ncnn.Net()
            # Use vulkan compute
            self.net.opt.use_vulkan_compute = True

            # Load model param and bin
            self.net.load_param(self.modelPath + ".param")
            self.net.load_model(self.modelPath + ".bin")
        elif method == "upscale_ncnn_py":
            self.net = UPSCALE(
            gpuid=self.gpuid,
            model_str=self.modelPath,
            num_threads=self.threads,
            scale=self.scale,
            tilesize=self.tilesize,
        )
          
        
    def hotUnload(self):
        self.model = None

    def hotReload(self):
        self._load()
    def NCNNImageMatFromNP(self, npArray: np.array):
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
        
        return nparray.clip(0, 255)

        
    def procNCNNVk(self, imageChunk):
        ex = self.net.create_extractor()
        frame = self.NCNNImageMatFromNP(imageChunk)
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
        return np.ascontiguousarray(frame, dtype=np.uint8)

    def Upscale(self, imageChunk):
        while self.net is None:
            sleep(1)
        if method == "ncnn_vulkan":
            return self.procNCNNVk(imageChunk)
        elif method == "upscale_ncnn_py":
            return self.net.process_bytes(imageChunk, self.width, self.height, 3)
        