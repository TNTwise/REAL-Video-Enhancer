import os
import math
import numpy as np
import cv2
import gc
import torch as torch
import torch.nn.functional as F
from time import sleep

from src.Util import (
    currentDirectory,
    modelsDirectory,
    printAndLog,
    check_bfloat16_support,
)

# tiling code permidently borrowed from https://github.com/chaiNNer-org/spandrel/issues/113#issuecomment-1907209731


class UpscalePytorch:
    """A class for upscaling images using PyTorch.

    Args:
        modelPath (str): The path to the model file.
        device (str, optional): The device to use for inference. Defaults to "default".
        tile_pad (int, optional): The padding size for tiles. Defaults to 10.
        precision (str, optional): The precision mode for the model. Defaults to "auto".
        width (int, optional): The width of the input image. Defaults to 1920.
        height (int, optional): The height of the input image. Defaults to 1080.
        backend (str, optional): The backend for inference. Defaults to "pytorch".
        trt_workspace_size (int, optional): The workspace size for TensorRT. Defaults to 0.
        trt_cache_dir (str, optional): The cache directory for TensorRT. Defaults to modelsDirectory().

    Attributes:
        tile_pad (int): The padding size for tiles.
        dtype (torch.dtype): The data type for the model.
        device (torch.device): The device used for inference.
        model (torch.nn.Module): The loaded model.
        width (int): The width of the input image.
        height (int): The height of the input image.
        scale (float): The scale factor of the model.

    Methods:
        handlePrecision(precision): Handles the precision mode for the model.
        loadModel(modelPath, dtype, device): Loads the model from file.
        bytesToFrame(frame): Converts bytes to a torch tensor.
        tensorToNPArray(image): Converts a torch tensor to a NumPy array.
        renderImage(image): Renders an image using the model.
        renderToNPArray(image): Renders an image and returns it as a NumPy array.
        renderImagesInDirectory(dir): Renders all images in a directory.
        getScale(): Returns the scale factor of the model.
        saveImage(image, fullOutputPathLocation): Saves an image to a file.
        renderTiledImage(image, tile_size): Renders a tiled image."""

    @torch.inference_mode()
    def __init__(
        self,
        modelPath: str,
        device="default",
        tile_pad: int = 10,
        precision: str = "auto",
        width: int = 1920,
        height: int = 1080,
        tilesize: int = 0,
        backend: str = "pytorch",
        # trt options
        trt_workspace_size: int = 0,
        trt_cache_dir: str = modelsDirectory(),
    ):
        if device == "default":
            if torch.cuda.is_available():
                device = torch.device(
                    "cuda", 0
                )  # 0 is the device index, may have to change later
            else:
                device = torch.device("cpu")
        else:
            decice = torch.device(device)
        printAndLog("Using device: " + str(device))
        self.tile_pad = tile_pad
        self.dtype = self.handlePrecision(precision)
        self.device = device
        self.videoWidth = width
        self.videoHeight = height
        self.tilesize = tilesize
        self.tile = [self.tilesize, self.tilesize]
        self.modelPath = modelPath
        self.backend = backend
        self.trt_cache_dir = trt_cache_dir
        self.trt_workspace_size = trt_workspace_size

        self._load()
    @torch.inference_mode()
    def _load(self):
        model = self.loadModel(modelPath=self.modelPath, device=self.device, dtype=self.dtype)



        match self.scale:
            case 1:
                modulo = 4
            case 2:
                modulo = 2
            case _:
                modulo = 1
        if all(t > 0 for t in self.tile):
            self.pad_w = (
                math.ceil(min(self.tile[0] + 2 * self.tile_pad, self.videoWidth) / modulo) * modulo
            )
            self.pad_h = (
                math.ceil(min(self.tile[1] + 2 * self.tile_pad, self.videoHheight) / modulo) * modulo
            )
        else:
            self.pad_w = self.videoWidth
            self.pad_h = self.videoHeight

        if self.backend == "tensorrt":
            import tensorrt as trt
            import torch_tensorrt

            trt_engine_path = os.path.join(
                os.path.realpath(self.trt_cache_dir),
                (
                    f"{os.path.basename(self.modelPath)}"
                    + f"_{self.pad_w}x{self.pad_h}"
                    + f"_{'fp16' if self.dtype == torch.float16 else 'fp32'}"
                    + f"_{torch.cuda.get_device_name(self.device)}"
                    + f"_trt-{trt.__version__}"
                    + (
                        f"_workspace-{self.trt_workspace_size}"
                        if self.trt_workspace_size > 0
                        else ""
                    )
                    + ".ts"
                ),
            )

            if not os.path.isfile(trt_engine_path):
                inputs = [
                    torch.zeros(
                        (1, 3, self.pad_h, self.pad_w),
                        dtype=self.dtype,
                        device=self.device,
                    )
                ]
                dummy_input_cpu_fp32 = [
                    torch.zeros(
                        (1, 3, 32, 32),
                        dtype=torch.float32,
                        device="cpu",
                    )
                ]

                module = torch.jit.trace(model.float().cpu(), dummy_input_cpu_fp32)
                module.to(device=self.device, dtype=self.dtype)
                module = torch_tensorrt.compile(
                    module,
                    ir="ts",
                    inputs=inputs,
                    enabled_precisions={self.dtype},
                    device=torch_tensorrt.Device(gpu_id=0),
                    workspace_size=self.trt_workspace_size,
                    truncate_long_and_double=True,
                    min_block_size=1,
                )

                torch.jit.save(module, trt_engine_path)

            model = torch.jit.load(trt_engine_path)

        self.model = model

    def handlePrecision(self, precision):
        if precision == "auto":
            return torch.float16 if check_bfloat16_support() else torch.float32
        if precision == "float32":
            return torch.float32
        if precision == "float16":
            return torch.float16

    def hotUnload(self):

        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
    def hotReload(self):
        self._load()
    @torch.inference_mode()
    def loadModel(
        self, modelPath: str, dtype: torch.dtype = torch.float32, device: str = "cuda"
    ) -> torch.nn.Module:
        from spandrel import ModelLoader, ImageModelDescriptor

        model = ModelLoader().load_from_file(modelPath)
        assert isinstance(model, ImageModelDescriptor)
        # get model attributes
        self.scale = model.scale

        model = model.model
        model.load_state_dict(model.state_dict(), assign=True)
        model.eval().to(self.device)
        if self.dtype == torch.float16:
            model.half()
        return model

    def bytesToFrame(self, frame):
        return (
            torch.frombuffer(frame, dtype=torch.uint8)
            .reshape(self.videoHeight, self.videoWidth, 3)
            .to(self.device, dtype=self.dtype)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .mul_(1 / 255)
        )

    def tensorToNPArray(self, image: torch.Tensor) -> np.array:
        image = image.squeeze(0).permute(1, 2, 0).float().mul(255).cpu().numpy()
        return image

    @torch.inference_mode()
    def renderImage(self, image: torch.Tensor) -> torch.Tensor:
        upscaledImage = self.model(image)
        return upscaledImage

    @torch.inference_mode()
    def renderToNPArray(self, image: torch.Tensor) -> torch.Tensor:
        while self.model is None:
            sleep(0)
        if self.tilesize == 0:
            output = self.renderImage(image)
        else:
            output = self.renderTiledImage(image)
        output = (
            output.squeeze(0)
            .permute(1, 2, 0)
            .float()
            .clamp(0.0, 1.0)
            .mul(255)
            .byte()
            .contiguous()
            .detach()
            .cpu()
            .numpy()
        )
        return output

    def getScale(self):
        return self.scale

    @torch.inference_mode()
    def renderTiledImage(
        self,
        img: torch.Tensor,
    ) -> torch.Tensor:
        scale = self.scale
        tile = self.tile
        tile_pad = self.tile_pad

        batch, channel, height, width = img.shape
        output_shape = (batch, channel, height * scale, width * scale)

        # start with black image
        output = img.new_zeros(output_shape)

        tiles_x = math.ceil(width / tile[0])
        tiles_y = math.ceil(height / tile[1])

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile[0]
                ofs_y = y * tile[1]

                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile[0], width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile[1], height)

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

                h, w = input_tile.shape[2:]
                input_tile = F.pad(
                    input_tile, (0, self.pad_w - w, 0, self.pad_h - h), "replicate"
                )

                # process tile
                output_tile = self.model(input_tile)

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
