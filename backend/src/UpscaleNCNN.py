import numpy as np
import os
from time import sleep
import math
import numpy as np

try:
    from upscale_ncnn_py import UPSCALE
    method = "upscale_ncnn_py"
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
        tilePad=10,
    ):
        # only import if necessary
        self.pad_w = tilePad
        self.pad_h = tilePad
        self.gpuid = gpuid
        self.modelPath = modelPath
        self.scale = scale
        self.tilesize = tilesize
        self.width = width
        self.height = height
        self.tilewidth = width
        self.tileheight = height
        if tilesize != 0:
            self.tilewidth = tilesize
            self.tile_size = tilesize
            self.tileheight = tilesize
        self.scale = scale
        self.threads = num_threads
        self.tilePad = tilePad
        self.tile_pad = tilePad
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
        self.net = None

    def hotReload(self):
        self._load()

    def NCNNImageMatFromNP(self, npArray: np.array):
        return ncnn.Mat.from_pixels(
            npArray,
            ncnn.Mat.PixelType.PIXEL_BGR,
            self.tilewidth,
            self.tileheight,
        )

    def NormalizeImage(self, mat, norm_vals):
        mean_vals = []
        mat.substract_mean_normalize(mean_vals, norm_vals)

    def ClampNPArray(self, nparray: np.array) -> np.array:
        nparray = np.clip(nparray, 0, 255)
        return nparray

    def procNCNNVk(self, frame: np.array) -> np.ascontiguousarray:
        ex = self.net.create_extractor()
        frame = self.ClampNPArray(frame)
        frame = self.NCNNImageMatFromNP(frame)
        # norm
        self.NormalizeImage(mat=frame, norm_vals=[1 / 255.0, 1 / 255.0, 1 / 255.0])
        # render frame
        ex.input("data", frame)
        ret, frame = ex.extract("output")

        # norm
        frame = np.array(frame)
        frame = frame.transpose(1, 2, 0) * 255
        frame = self.ClampNPArray(frame)
        return np.ascontiguousarray(frame, dtype=np.uint8)

    def Upscale(self, imageChunk):
        while self.net is None:
            sleep(1)
        if method == "ncnn_vulkan":
            frame = np.ascontiguousarray(np.frombuffer(imageChunk, dtype=np.uint8))
            if self.tilesize == 0:
                return self.procNCNNVk(frame)
            else:
                return self.renderTiledImage(frame)
        elif method == "upscale_ncnn_py":
            return self.net.process_bytes(imageChunk, self.width, self.height, 3)

    def renderTiledImage(self, img: np.ndarray):
        raise NotImplementedError("Tile rendering not implemented for default ncnn fallback, please install vcredlist from https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170")
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.

        Modified from: https://github.com/ata4/esrgan-launcher
        """
        img = img.reshape(3, self.height, self.width)
        channel, height, width = img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (channel, output_height, output_width)

        # start with black image
        self.output = np.zeros(output_shape, dtype=img.dtype)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = img[:, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                output_tile = self.procNCNNVk(input_tile).transpose(2, 0,1)
                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]
        return self.output