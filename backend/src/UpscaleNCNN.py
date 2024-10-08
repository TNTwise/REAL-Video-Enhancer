import numpy as np
import os
from time import sleep

from ncnn_vulkan import ncnn

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

import numpy as np

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
        self.net = ncnn.Net()
        # Use vulkan compute
        self.net.opt.use_vulkan_compute = True

        # Load model param and bin
        self.load_param(self.modelPath + ".param")
        self.load_model(self.modelPath + ".bin")

        self.ex = self.net.create_extractor()
        
    def hotUnload(self):
        self.model = None

    def hotReload(self):
        self._load()

    def Upscale(self, imageChunk):
        while self.model is None:
            sleep(1)
        img_in = np.frombuffer(imageChunk, dtype=np.uint8).reshape(1080, 1920, 3)
        mat_in = ncnn.Mat.from_pixels(
            img_in,
            ncnn.Mat.PixelType.PIXEL_BGR,
            img_in.shape[1],
            img_in.shape[0]
        )
        mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)
        try:
            # Make sure the input and output names match the param file
            self.ex.input("data", mat_in)
            ret, mat_out = self.ex.extract("output")
            out = np.array(mat_out)

            # Transpose the output from `c, h, w` to `h, w, c` and put it back in 0-255 range
            return out.transpose(1, 2, 0).__mul__(255.0).astype(np.uint8).tobytes()

        except:
            ncnn.destroy_gpu_instance()
