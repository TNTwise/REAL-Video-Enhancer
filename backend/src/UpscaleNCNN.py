import numpy as np
import os
from time import sleep

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
        self._load()

    def _load(self):
        self.model = UPSCALE(
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

    def Upscale(self, imageChunk):
        while self.model is None:
            sleep(1)
        output = self.model.process_bytes(imageChunk, self.width, self.height, 3)
        return np.ascontiguousarray(
            np.frombuffer(
                output,
                dtype=np.uint8,
            ).reshape(self.height * self.scale, self.width * self.scale, 3)
        )
