import pathlib
import sys
from time import sleep

import cv2
import numpy as np

from src.logic.handlers.backend.backend_handler import BackendHandler
from src.logic.handlers.backend.ncnn_handler import NCNNHandler
from src.logic.render.methods.interpolate.interpolate_method_base import (
    InterpolateMethodBase,
)
from src.schemas.domain import Frame, InterpolateModel
from src.schemas.domain.video_info import InputVideoInfo


class Rife:
    def __init__(
        self,
        wrapped,
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
        self.wrapped = wrapped
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
        self._rife_object = self.wrapped.RifeWrapped(
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
            modeldir_str = self.wrapped.StringType()
            if sys.platform in ("win32", "cygwin"):
                modeldir_str.wstr = self.wrapped.new_wstr_p()
                self.wrapped.wstr_p_assign(modeldir_str.wstr, str(model_dir))
            else:
                modeldir_str.str = self.wrapped.new_str_p()
                self.wrapped.str_p_assign(modeldir_str.str, str(model_dir))

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
            self.raw_in_image0 = self.wrapped.Image(
                self.image0_bytes, self.width, self.height, self.channels
            )
        self.image1_bytes = bytearray(image1_bytes)

        self.raw_in_image1 = self.wrapped.Image(
            self.image1_bytes, self.width, self.height, self.channels
        )

        self._rife_object.process(
            self.raw_in_image0, self.raw_in_image1, timestep, self.raw_out_image
        )

        if timestep == self.max_timestep:
            self.image0_bytes = self.image1_bytes
            self.raw_in_image0 = self.raw_in_image1
        return bytes(self.output_bytes)


class InterpolateNCNN(InterpolateMethodBase):
    def __init__(
        self,
        backend_handler: BackendHandler,
        interpolate_model: InterpolateModel,
        input_video_info: InputVideoInfo,
    ):
        assert isinstance(backend_handler, NCNNHandler)
        # TODO: Create a model for Device, had attribute id and name
        self._gpu_id = 0
        # TODO: Implement paused feature
        self._paused = False
        self._backend_hander = backend_handler
        self._interpolate_model = interpolate_model
        self._input_video_info = input_video_info
        self.frame0 = None
        self._save_counter = 0
        self._output_dir = pathlib.Path("/tmp/debug_frames")
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _load(self):
        max_timestep = (
            self._interpolate_model.interpolate_factor - 1
        ) / self._interpolate_model.interpolate_factor
        self.render = Rife(
            wrapped=self._backend_hander.get_rife().wrapped,
            gpuid=self._gpu_id,
            num_threads=1,
            model=self._interpolate_model.file_path,
            uhd_mode=False,
            channels=3,
            height=self._input_video_info.height,
            width=self._input_video_info.width,
            max_timestep=max_timestep,
        )
        # device = ncnn.get_gpu_device(self.gpuid).info().device_name()

    def _debug_save_frame(ret_frame):
        np_frame = ret_frame.get_frame_np()
        if np_frame.dtype == np.uint16:
            np_frame = (
                np.clip(np_frame.astype(np.float32) / 65535.0, 0, 1) * 255
            ).astype(np.uint8)
        bgr = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
        self._save_counter += 1
        cv2.imwrite(
            str(self._output_dir / f"frame_{self._save_counter:06d}.jpg"),
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

    def process_frame(
        self,
        frame: Frame,
        transition=False,
    ):
        if self.frame0 is None:
            self.frame0 = frame.get_frame_bytes()
            return
        if transition:
            self.render.process_bytes(
                self.frame0, frame.get_frame_bytes(), 0
            )  # get the cache to skip to next frame
            self.frame0 = frame

            for n in range(self._interpolate_model.interpolate_factor - 1):
                yield frame
            return
        for n in range(self._interpolate_model.interpolate_factor - 1):
            while self._paused:
                sleep(1)
            timestep = (n + 1) * 1.0 / (self._interpolate_model.interpolate_factor)
            interp_frame = self.render.process_bytes(
                self.frame0, frame.get_frame_bytes(), timestep
            )
            retFrame = Frame(
                self._input_video_info.width,
                self._input_video_info.height,
            )
            retFrame.set_frame_bytes(interp_frame)

            yield retFrame
        self.frame0 = frame
