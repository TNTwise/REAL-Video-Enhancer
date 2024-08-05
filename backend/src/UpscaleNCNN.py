import numpy as np
import os

from upscale_ncnn_py import UPSCALE


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
        modelPath: str,
        num_threads: int,
        scale: int,
        gpuid: int = 0,
        width: int = 1920,
        height: int = 1080,
    ):
        # only import if necessary

        self.model = UPSCALE(
            gpuid=gpuid, model_str=modelPath, num_threads=num_threads, scale=scale
        )
        self.width = width
        self.height = height
        self.scale = scale

    def Upscale(self, imageChunk):
        output = self.model.process_bytes(imageChunk, self.width, self.height, 3)
        return np.ascontiguousarray(
            np.frombuffer(
                output,
                dtype=np.uint8,
            ).reshape(self.height * self.scale, self.width * self.scale, 3)
        )
