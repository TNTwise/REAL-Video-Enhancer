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
            self.tileheight = tilesize
        self.scale = scale
        self.threads = num_threads
        self.tilePad = tilePad
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
        return nparray.clip(0, 255)

    def procNCNNVk(self, imageChunk) -> np.ascontiguousarray:
        ex = self.net.create_extractor()
        frame = self.NCNNImageMatFromNP(imageChunk)
        # norm
        self.NormalizeImage(mat=frame, norm_vals=[1 / 255.0, 1 / 255.0, 1 / 255.0])
        # render frame
        ex.input("data", frame)
        ret, frame = ex.extract("output")

        # norm
        self.NormalizeImage(mat=frame, norm_vals=[255.0, 255.0, 255.0])
        frame = np.array(frame)
        frame = self.ClampNPArray(frame)
        return frame

    def Upscale(self, imageChunk):
        while self.net is None:
            sleep(1)
        if method == "ncnn_vulkan":
            if self.tilesize == 0:
                return self.procNCNNVk(imageChunk).transpose(1, 2, 0)
            else:
                npArray = (
                    np.frombuffer(imageChunk, dtype=np.uint8)
                    .reshape(self.height, self.width, 3)
                    .transpose(2, 0, 1)
                )[np.newaxis, ...]
                return self.renderTiledImage(npArray)
        elif method == "upscale_ncnn_py":
            return self.net.process_bytes(imageChunk, self.width, self.height, 3)

    def renderTiledImage(
        self,
        img
    ):
        raise NotImplementedError("Tiling is not supported on this configuration!")
        scale = self.scale
        tile = self.tilesize
        tile_pad = self.tilePad

        batch, channel, height, width = img.shape
        output_shape = (batch, channel, height * scale, width * scale)

        # start with black image
        output = np.zeros(output_shape, dtype=img.dtype)

        tiles_x = math.ceil(width / tile)
        tiles_y = math.ceil(height / tile)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile
                ofs_y = y * tile

                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y

                input_tile = img[
                    :,
                    :,
                    input_start_y_pad:input_end_y_pad,
                    input_start_x_pad:input_end_x_pad,
                ]

                input_tile = np.pad(input_tile, 
                           ((0, 0), (0, 0), (0, self.pad_h), (0, self.pad_w)), 
                           mode='edge')

                # process tile
                output_tile = self.procNCNNVk(input_tile.squeeze(axis=0))[np.newaxis, ...]
                print(output_tile.shape)

                output_tile = output_tile[:, :, : h * scale, : w * scale]

                # output tile area on total image
                output_start_x = input_start_x * scale
                output_end_x = input_end_x * scale
                output_start_y = input_start_y * scale
                output_end_y = input_end_y * scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * scale
                output_end_x_tile = output_start_x_tile + input_tile_width * scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * scale
                output_end_y_tile = output_start_y_tile + input_tile_height * scale

                # put tile into output image
                output[
                    :, :, output_start_y:output_end_y, output_start_x:output_end_x
                ] = output_tile[
                    :,
                    :,
                    output_start_y_tile:output_end_y_tile,
                    output_start_x_tile:output_end_x_tile,
                ]

        return output
        
