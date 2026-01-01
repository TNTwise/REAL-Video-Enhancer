from rife_ncnn_vulkan_python import wrapped
from time import sleep

# built-in imports
import pathlib
import sys
from ..utils.Util import suppress_stdout_stderr
from ..utils.Frame import Frame

# third-party imports
import numpy as np
import ncnn


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
        self.image1_bytes = None
        self.raw_in_image1 = None
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

    def patch_pause(self):
        """
        Used in instances where the scene change is active, and the frame needs to be uncached.
        """
        self.image0_bytes = self.image1_bytes
        self.raw_in_image0 = self.raw_in_image1

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
        self.image1_bytes = bytearray(image1_bytes)

        self.raw_in_image1 = wrapped.Image(
            self.image1_bytes, self.width, self.height, self.channels
        )

        self._rife_object.process(
            self.raw_in_image0, self.raw_in_image1, timestep, self.raw_out_image
        )

        if timestep == self.max_timestep:
            self.image0_bytes = self.image1_bytes
            self.raw_in_image0 = self.raw_in_image1
        return bytes(self.output_bytes)


class InterpolateRIFENCNN:
    def __init__(
        self,
        interpolateModelPath: str,
        width: int = 1920,
        height: int = 1080,
        threads: int = 1,
        gpuid: int = 0,
        max_timestep: int = 1,
        interpolateFactor: int = 2,
        hdr_mode: bool = False,
    ):
        self.max_timestep = max_timestep
        self.interpolateFactor = interpolateFactor
        self.interpolateModelPath = interpolateModelPath
        self.width = width
        self.height = height
        self.gpuid = gpuid
        self.threads = threads
        self.paused = False
        self.backend = "ncnn"
        self.frame0 = None
        self.hdr_mode = hdr_mode 
        self._load()

    def _load(self):
        with suppress_stdout_stderr():
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
            device = ncnn.get_gpu_device(self.gpuid).info().device_name()
        print("Using GPU:", device)

    def hotUnload(self):
        self.paused = True
        self.render.patch_pause()

    def hotReload(self):
        self.paused = False

    def __call__(
        self,
        img1: Frame,
        transition=False,
    ):
        if self.frame0 is None:
            self.frame0 = img1.get_frame_bytes()
            return
        if transition:
            self.render.process_bytes(
                self.frame0, img1.get_frame_bytes(), self.max_timestep
            )  # get the cache to skip to next frame
            self.frame0 = img1
            
            for n in range(self.interpolateFactor - 1):
                yield img1
            return
        for n in range(self.interpolateFactor - 1):
            while self.paused:
                sleep(1)
            timestep = (n + 1) * 1.0 / (self.interpolateFactor)
            frame = self.render.process_bytes(self.frame0, img1.get_frame_bytes(), timestep)
            retFrame = Frame(self.backend, self.width, self.height, img1.device, gpu_id=img1.gpu_id, hdr_mode=self.hdr_mode, dtype=img1.dtype)
            retFrame.set_frame_bytes(frame)
            yield retFrame
        self.frame0 = img1