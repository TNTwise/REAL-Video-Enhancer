from rife_ncnn_vulkan_python import wrapped
from time import sleep

# built-in imports
import importlib
import pathlib
import sys

# third-party imports
from PIL import Image
import numpy as np
import cv2


class Rife:
    def __init__(
        self,
        gpuid: int = -1,
        model: str = "rife-v2.3",
        scale: int = 2,
        tta_mode: bool = False,
        tta_temporal_mode: bool = False,
        uhd_mode: bool = False,
        num_threads: int = 1,
        channels: int = 3,
        width: int = 1920,
        height: int = 1080,
        max_timestep: float = 1.0,
    ):
        self.image0_bytes = None
        self.raw_in_image0 = None
        self.channels = None
        self.height = height
        self.width = width
        self.channels = channels
        self.max_timestep = max_timestep
        self.output_bytes = bytearray(width * height * channels)
        self.raw_out_image = wrapped.Image(
            self.output_bytes, self.width, self.height, self.channels
        )
        # scale must be a power of 2
        if (scale & (scale - 1)) == 0:
            self.scale = scale
        else:
            raise ValueError("scale should be a power of 2")

        # determine if rife-v2 is used
        rife_v2 = ("rife-v2" in model) or ("rife-v3" in model)
        rife_v4 = "rife-v4" in model or "rife4" in model or "rife-4" in model
        padding = 32
        if ("rife-v4.25" in model) or ("rife-v4.26" in model):
            padding = 64

        # create raw RIFE wrapper object
        self._rife_object = wrapped.RifeWrapped(
            gpuid,
            tta_mode,
            tta_temporal_mode,
            uhd_mode,
            num_threads,
            rife_v2,
            rife_v4,
            padding,
        )
        self._load(model)

    def _load(self, model: str, model_dir: pathlib.Path = None):
        # if model_dir is not specified
        if model_dir is None:
            model_dir = pathlib.Path(model)
            if not model_dir.is_absolute() and not model_dir.is_dir():
                model_dir = pathlib.Path(__file__).parent / "models" / model

        # if the model_dir is specified and exists
        if model_dir.exists():
            modeldir_str = wrapped.StringType()
            if sys.platform in ("win32", "cygwin"):
                modeldir_str.wstr = wrapped.new_wstr_p()
                wrapped.wstr_p_assign(modeldir_str.wstr, str(model_dir))
            else:
                modeldir_str.str = wrapped.new_str_p()
                wrapped.str_p_assign(modeldir_str.str, str(model_dir))

            self._rife_object.load(modeldir_str)

        # if no model_dir is specified but doesn't exist
        else:
            raise FileNotFoundError(f"{model_dir} not found")

    def process(self, image0: Image, image1: Image, timestep: float = 0.5) -> Image:
        # Return the image immediately instead of doing the copy in the upstream part which cause black output problems
        # The reason is that the upstream code use ncnn::Mat::operator=(const Mat& m) does a reference copy which won't
        # change our OutImage data.
        if timestep == 0.0:
            return image0
        elif timestep == 1.0:
            return image1

        image0_bytes = bytearray(image0.tobytes())
        image1_bytes = bytearray(image1.tobytes())
        channels = int(len(image0_bytes) / (image0.width * image0.height))
        output_bytes = bytearray(len(image0_bytes))

        # convert image bytes into ncnn::Mat Image
        raw_in_image0 = wrapped.Image(
            image0_bytes, image0.width, image0.height, channels
        )
        raw_in_image1 = wrapped.Image(
            image1_bytes, image1.width, image1.height, channels
        )
        raw_out_image = wrapped.Image(
            output_bytes, image0.width, image0.height, channels
        )

        self._rife_object.process(raw_in_image0, raw_in_image1, timestep, raw_out_image)
        return Image.frombytes(
            image0.mode, (image0.width, image0.height), bytes(output_bytes)
        )

    def process_cv2(
        self, image0: np.ndarray, image1: np.ndarray, timestep: float = 0.5
    ) -> np.ndarray:
        if timestep == 0.0:
            return image0
        elif timestep == 1.0:
            return image1

        image0_bytes = bytearray(image0.tobytes())
        image1_bytes = bytearray(image1.tobytes())

        self.channels = int(len(image0_bytes) / (image0.shape[1] * image0.shape[0]))
        self.output_bytes = bytearray(len(image0_bytes))

        # convert image bytes into ncnn::Mat Image
        raw_in_image0 = wrapped.Image(
            image0_bytes, image0.shape[1], image0.shape[0], self.channels
        )
        raw_in_image1 = wrapped.Image(
            image1_bytes, image0.shape[1], image0.shape[0], self.channels
        )
        raw_out_image = wrapped.Image(
            self.output_bytes, image0.shape[1], image0.shape[0], self.channels
        )

        self._rife_object.process(raw_in_image0, raw_in_image1, timestep, raw_out_image)

        return np.frombuffer(self.output_bytes, dtype=np.uint8).reshape(
            image0.shape[0], image0.shape[1], self.channels
        )

    def uncache_frame(self):
        """
        Used in instances where the scene change is active, and the frame needs to be uncached.
        """
        self.image0_bytes = None
        self.raw_in_image0 = None

    def process_bytes(
        self, image0_bytes, image1_bytes, timestep: float = 0.5
    ) -> np.ndarray:
        # print(timestep)
        if timestep == 0.0:
            return image0_bytes
        elif timestep == 1.0:
            return image1_bytes

        if self.image0_bytes is None:
            self.image0_bytes = bytearray(image0_bytes)
            self.raw_in_image0 = wrapped.Image(
                self.image0_bytes, self.width, self.height, self.channels
            )
        image1_bytes = bytearray(image1_bytes)

        raw_in_image1 = wrapped.Image(
            image1_bytes, self.width, self.height, self.channels
        )

        self._rife_object.process(
            self.raw_in_image0, raw_in_image1, timestep, self.raw_out_image
        )
        if timestep == self.max_timestep:
            self.image0_bytes = image1_bytes
            self.raw_in_image0 = raw_in_image1

        return bytes(self.output_bytes)

    def process_fast(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        timestep: float = 0.5,
        shape: tuple = None,
        channels: int = 3,
    ) -> np.ndarray:
        """
        An attempt at a faster implementation for NCNN that should speed it up significantly through better caching methods.

        :param image0: The first image to be processed.
        :param image1: The second image to be processed.
        :param timestep: The timestep value for the interpolation.
        :param shape: The shape of the images.
        :param channels: The number of channels in the images.

        :return: The processed image, format: np.ndarray.
        """

        if timestep == 0.0:
            return np.array(image0)
        elif timestep == 1.0:
            return np.array(image1)

        if self.height == None:
            if shape is None:
                self.height, self.width, self.channels = image0.shape
            else:
                self.height, self.width = shape

        image1_bytes = bytearray(image1.tobytes())
        raw_in_image1 = wrapped.Image(image1_bytes, self.width, self.height, channels)

        if self.image0_bytes is None:
            self.image0_bytes = bytearray(image0.tobytes())
            self.output_bytes = bytearray(len(self.image0_bytes))

        raw_in_image0 = wrapped.Image(
            self.image0_bytes, self.width, self.height, channels
        )

        raw_out_image = wrapped.Image(
            self.output_bytes, self.width, self.height, channels
        )

        self._rife_object.process(raw_in_image0, raw_in_image1, timestep, raw_out_image)

        self.image0_bytes = image1_bytes

        return np.frombuffer(self.output_bytes, dtype=np.uint8).reshape(
            self.height, self.width, self.channels
        )

    def process_fast_torch(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        timestep: float = 0.5,
        shape: tuple = None,
        channels: int = 3,
    ) -> np.ndarray:
        """
        An attempt at a faster implementation for NCNN that should speed it up significantly through better caching methods.

        :param image0: The first image to be processed.
        :param image1: The second image to be processed.
        :param timestep: The timestep value for the interpolation.
        :param shape: The shape of the images.
        :param channels: The number of channels in the images.

        :return: The processed image, format: torch.uint8
        """
        if self.height is None:
            if shape is None:
                self.height, self.width, self.channels = image0.shape
            else:
                self.height, self.width = shape

        image1_bytes = bytearray(image1)
        raw_in_image1 = wrapped.Image(
            image1_bytes, self.width, self.height, self.channels
        )

        if self.image0_bytes is None:
            self.image0_bytes = bytearray(image0)
            raw_in_image0 = wrapped.Image(
                self.image0_bytes, self.width, self.height, self.channels
            )
            self.output_bytes = bytearray(len(self.image0_bytes))
        else:
            raw_in_image0 = wrapped.Image(
                self.image0_bytes, self.width, self.height, self.channels
            )

        raw_out_image = wrapped.Image(
            self.output_bytes, self.width, self.height, self.channels
        )

        self._rife_object.process(raw_in_image0, raw_in_image1, timestep, raw_out_image)

        self.image0_bytes = image1_bytes

        return torch.frombuffer(self.output_bytes, dtype=torch.uint8).reshape(
            self.height, self.width, self.channels
        )


class RIFE(Rife): ...


class InterpolateRIFENCNN:
    def __init__(
        self,
        interpolateModelPath: str,
        width: int = 1920,
        height: int = 1080,
        threads: int = 1,
        gpuid: int = 0,
        max_timestep: int = 1,
    ):
        self.max_timestep = max_timestep
        self.interpolateModelPath = interpolateModelPath
        self.width = width
        self.height = height
        self.gpuid = gpuid
        self.threads = threads
        self._load()

    def _load(self):
        self.render = Rife(
            gpuid=self.gpuid,
            num_threads=self.threads,
            model=self.interpolateModelPath,
            uhd_mode=False,
            channels=3,
            height=self.height,
            width=self.width,
            max_timestep=self.max_timestep,
        )

    def hotUnload(self):
        self.render._rife_object.__swig_destroy__(self.render._rife_object)
        self.render = None

    def hotReload(self):
        self._load()

    def process(self, img0, img1, timestep) -> bytes:
        while self.render is None:
            sleep(1)
        frame = self.render.process_bytes(img0, img1, timestep)
        return frame

    def normFrame(self, frame: bytes):
        return frame
        frame = bytearray(frame)
        frame = wrapped.Image(frame, self.width, self.height, 3)
        return frame

    def uncacheFrame(self):
        return
        self.render.uncache_frame()
