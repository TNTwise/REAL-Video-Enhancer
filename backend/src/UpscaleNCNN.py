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

        self.gpuid = gpuid
        self.modelPath = modelPath
        self.scale = scale
        self.tilesize = tilesize
        self.width = width
        self.height = height
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
            if self.tilesize == 0:
                return self.procNCNNVk(imageChunk)
            else:
                npArray = (
                    np.frombuffer(imageChunk, dtype=np.uint8)
                    .reshape(self.height, self.width, 3)
                    .transpose(2, 0, 1)
                )
                return self.upscaleTiledImage(npArray)
        elif method == "upscale_ncnn_py":
            return self.net.process_bytes(imageChunk, self.width, self.height, 3)

    def upscaleTiledImage(self, img: np.array):
        batch, channel, height, width = img.shape
        output_shape = (batch, channel, height * self.scale, width * self.scale)

        # Start with a black image
        output = np.zeros(output_shape, dtype=img.dtype)

        tiles_x = math.ceil(width / self.tilesize[0])
        tiles_y = math.ceil(height / self.tilesize[1])

        # Loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # Extract tile from input image
                ofs_x = x * self.tilesize[0]
                ofs_y = y * self.tilesize[1]

                # Input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tilesize[0], width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tilesize[1], height)

                # Input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tilePad, 0)
                input_end_x_pad = min(input_end_x + self.tilePad, width)
                input_start_y_pad = max(input_start_y - self.tilePad, 0)
                input_end_y_pad = min(input_end_y + self.tilePad, height)

                # Input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y

                # Extract the input tile with padding
                input_tile = img[
                    :,
                    :,
                    input_start_y_pad:input_end_y_pad,
                    input_start_x_pad:input_end_x_pad,
                ]

                # Pad the input tile
                h, w = input_tile.shape[2:]
                pad_h = max(0, self.tilePad - h)
                pad_w = max(0, self.tilePad - w)
                input_tile = np.pad(
                    input_tile,
                    ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                    mode="edge",
                )

                # Process tile using the model (assuming model is a function that can process numpy arrays)
                output_tile = self.procNCNNVk(input_tile)

                # Crop output tile to the expected size
                output_tile = output_tile[:, :, : h * self.scale, : w * self.scale]

                # Output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # Output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # Put tile into output image
                output[
                    :, :, output_start_y:output_end_y, output_start_x:output_end_x
                ] = output_tile[
                    :,
                    :,
                    output_start_y_tile:output_end_y_tile,
                    output_start_x_tile:output_end_x_tile,
                ]

        return output
