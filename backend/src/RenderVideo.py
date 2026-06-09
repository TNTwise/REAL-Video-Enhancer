import math
import os
import sys
import threading
import traceback
from multiprocessing import shared_memory
from threading import Thread
from time import sleep

from backend.src.schemas import RenderSettings, Setting
import cv2
import numpy as np

from src.services.ffmpeg_service import ReadBuffer, WriteBuffer
from src.InformationWriteOut import InformationWriteOut
from src.utils.LogConfig import get_logger
from src.utils.SceneDetect import SceneDetect
from src.utils.BorderDetect import BorderDetect
from src.services.video_info_service import VideoInfo

logger = get_logger(__name__)

class Render:
    def __init__(
        self,
        render_settings: RenderSettings,
        settings: Setting,
        video_info: VideoInfo,
        border_detect: BorderDetect,
        read_buffer: ReadBuffer,
        write_buffer: WriteBuffer,
    ):
        self.render_settings = render_settings
        self.settings = settings
        self.video_info = video_info
        self.border_detect = border_detect
        self.read_buffer = read_buffer
        self.write_buffer = write_buffer

        # TODO (bug #1): 'inputFile' is used but not defined in the __init__ params or passed from RenderSettings — will raise NameError at runtime.
        # TODO (bug #2): 'outputFile' is not defined — same root cause as bug #1.
        # TODO (bug #3): 'backend' is not defined — same root cause.
        # TODO (bug #4): 'device' is not defined — same root cause.
        # TODO (bug #5): 'precision' is not defined — same root cause.
        # TODO (bug #8): 'crf', 'video_encoder', 'audio_encoder', 'subtitle_encoder' are passed to FFmpegWrite but never bound on self or received as params.
        # TODO (bug #9): 'color_space', 'color_primaries', 'color_transfer', 'input_pix_fmt' are passed to FFmpegRead but never bound.
        # max timestep is a hack to make sure ncnn cache frames too early, and ncnn breaks if i modify the code at all so ig this is what we are doing
        # also used to help with performace and caching
        # must use ceilInterpolateFactor so the last timestep matches exactly
        self.maxTimestep = (self.ceilInterpolateFactor - 1) / self.ceilInterpolateFactor

        # self.setupRender = self.returnFrame  # set it to not convert the bytes to array by default, and just pass chunk through

        logger.info("Using backend: %s", self.backend)
        # upscale has to be called first to get the scale of the upscale model
        if render_settings.upscale_:
            self.setupUpscale()
            self.upscaleOption.hotUnload()  # unload model to free up memory for trt enging building
            logger.info("Using Upscaling Model: %s", self.upscaleModel)
        else:
            self.upscaleTimes = 1  # if no upscaling, it will default to 1
            self.modelScale = 1

        if render_settings.extra_restoration_models:
            for model in render_settings.extra_restoration_models:
                extraRestoration = self.setupExtraRestoration(model)
                if extraRestoration:
                    logger.info("Using Extra Restoration Model: %s", model)
                    self.extraRestorationModels.append(extraRestoration)
                    extraRestoration.hotUnload()  # unload model to free up memory for trt enging building

        if render_settings.interpolate_model:
            self.setupInterpolate()
            logger.info("Using Interpolation Model: %s", self.interpolateModel)

        if render_settings.upscale_model:
            self.upscaleOption.hotReload()

        for extraRestoration in self.extraRestorationModels:
            extraRestoration.hotReload()

    def write_bytes_to_cv2_frame_debug(self, frame):
        # Convert the byte array to a numpy array
        frame_array = np.frombuffer(frame, dtype=np.uint8)
        # Reshape the array to the correct dimensions
        frame_array = frame_array.reshape((self.height, self.width, 3))
        # Convert the BGR image to RGB
        frame_array = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
        cv2.imwrite("frame.jpg", frame_array)

    def render(self):
        frames_rendered = 0
        """
        from viztracer import VizTracer
        tracer = VizTracer()
        tracer.start()
        from pyinstrument import Profiler
        profiler = Profiler()
        profiler.start()
        """
        frame = self.readBuffer.get()
        while frame:
            if self.informationHandler.get_is_paused():
                sleep(1)

            for extraRestoration in self.extraRestorationModels:
                frame = extraRestoration(frame)

            if self.interpolateModel:
                sceneDetect = self.sceneDetect.detect(frame)
                interpolated_frames = self.interpolateOption(
                    img1=frame,
                    transition=sceneDetect,
                )

                for interpolated_frame in interpolated_frames:
                    if self.upscaleModel:
                        interpolated_frame = self.upscaleOption(interpolated_frame)
                    if self.override_upscale_scale:
                        interpolated_frame = interpolated_frame.resize_frame_optimal(
                            new_width=self.width * self.override_upscale_scale,
                            new_height=self.height * self.override_upscale_scale,
                        )

                    self.informationHandler.update(
                        interpolated_frame.get_frame_bytes(clear_cache=True)
                    )
                    self.writeBuffer.writeQueue.put(
                        interpolated_frame.get_frame_bytes(clear_cache=True)
                    )

            if self.upscaleModel:
                frame = self.upscaleOption(frame)

            if self.override_upscale_scale:
                frame = frame.resize_frame_optimal(
                    self.width * self.override_upscale_scale,
                    self.height * self.override_upscale_scale,
                )

            self.informationHandler.update(frame.get_frame_bytes(clear_cache=True))

            self.writeBuffer.writeQueue.put(frame.get_frame_bytes())
            frames_rendered += int(self.ceilInterpolateFactor)

            # grab new frame
            frame = self.readBuffer.get()

        self.informationHandler.stopWriting()
        self.writeBuffer.writeQueue.put(None)
        """
        tracer.stop()
        tracer.save()
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
        """

    def upscalePytorchObject(self, modelPath=None):
        from .services.pytorch.UpscaleTorch import UpscalePytorch

        return UpscalePytorch(
            modelPath,
            device=self.device,
            precision=self.precision,
            width=self.width,
            height=self.height,
            backend=self.backend,
            tilesize=self.tilesize,
            gpu_id=self.pytorch_gpu_id,
            trt_optimization_level=self.trt_optimization_level,
            hdr_mode=self.hdr_mode,
            trt_static_shape=not self.trt_dynamic_shapes,
        )

    def upscaleNCNNObject(self, scale=None, modelPath=None):
        from .services.ncnn.UpscaleNCNN import UpscaleNCNN

        path, last_folder = os.path.split(modelPath)
        modelPath = os.path.join(path, last_folder, last_folder)
        return UpscaleNCNN(
            modelPath=modelPath,
            num_threads=1,
            scale=self.upscaleTimes if scale is None else scale,
            gpuid=self.ncnn_gpu_id,  # might have this be a setting
            width=self.width,
            height=self.height,
            tilesize=self.tilesize,
        )

    def upscaleONNXObject(self, scale=None, modelPath=None):
        from .services.onnx.UpscaleONNX import UpscaleONNX

        return UpscaleONNX(
            modelPath=modelPath,
            deviceID=self.pytorch_gpu_id,
            precision=self.precision,
            width=self.width,
            height=self.height,
            scale=self.upscaleTimes if scale is None else scale,
            hdr_mode=self.hdr_mode,
        )

    def setupExtraRestoration(self, modelPath):
        logger.info("Setting up Extra Restoration")
        if self.backend == "pytorch" or self.backend == "tensorrt":
            return self.upscalePytorchObject(modelPath)

        if self.backend == "ncnn":
            return self.upscaleNCNNObject(scale=1, modelPath=modelPath)

    def setupUpscale(self):
        logger.info("Setting up Upscale")
        if self.backend == "pytorch" or self.backend == "tensorrt":
            self.upscaleOption = self.upscalePytorchObject(self.upscaleModel)
            self.modelScale = self.upscaleOption.getScale()

        if self.backend == "ncnn":
            from .services.ncnn.UpscaleNCNN import getNCNNScale

            self.modelScale = getNCNNScale(modelPath=self.upscaleModel)

            self.upscaleOption = self.upscaleNCNNObject(
                scale=self.modelScale, modelPath=self.upscaleModel
            )

        if self.backend == "directml":  # i dont want to work with this shit
            from .services.onnx.UpscaleONNX import UpscaleONNX

            self.modelScale = UpscaleONNX.getModelScale(self.upscaleModel)

            self.upscaleOption = UpscaleONNX(
                modelPath=self.upscaleModel,
                precision=self.precision,
                width=self.width,
                height=self.height,
                scale=self.modelScale,
            )
        self.upscaleTimes = (
            self.modelScale
            if not self.override_upscale_scale
            else self.override_upscale_scale
        )

    def setupInterpolate(self):
        logger.info("Setting up Interpolation")
        self.sceneDetect = SceneDetect(
            sceneChangeMethod=self.sceneDetectMethod,
            sceneChangeSensitivity=self.sceneDetectSensitivty,
            width=self.width,
            height=self.height,
            model_path=self.sceneDetectModelPath,
            model_backend=self.backend,
            model_dtype=self.precision,
            model_device=self.device,
            model_gpu_id=self.pytorch_gpu_id
            if self.backend in ["pytorch", "tensorrt"]
            else self.ncnn_gpu_id,
        )
        if self.sceneDetectMethod != "none":
            logger.info("Scene Detection Enabled")

        else:
            logger.info("Scene Detection Disabled")

        if self.backend == "ncnn":
            from .services.ncnn.InterpolateNCNN import InterpolateRIFENCNN

            self.interpolateOption = InterpolateRIFENCNN(
                interpolateModelPath=self.interpolateModel,
                width=self.width,
                height=self.height,
                gpuid=self.ncnn_gpu_id,
                max_timestep=self.maxTimestep,
                interpolateFactor=self.ceilInterpolateFactor,
                hdr_mode=self.hdr_mode,
            )

        if self.backend == "pytorch" or self.backend == "tensorrt":
            from .services.pytorch.InterpolateTorch import InterpolateFactory

            self.interpolateOption = InterpolateFactory.build_interpolation_method(
                self.interpolateModel,
                self.backend,
            )(
                modelPath=self.interpolateModel,
                ceilInterpolateFactor=self.ceilInterpolateFactor,
                width=self.width,
                height=self.height,
                device=self.device,
                dtype=self.precision,
                backend=self.backend,
                gpu_id=self.pytorch_gpu_id,
                UHDMode=self.UHD_mode,
                trt_optimization_level=self.trt_optimization_level,
                ensemble=self.ensemble,
                dynamicScaledOpticalFlow=self.dynamic_scaled_optical_flow,
                max_timestep=self.maxTimestep,
                hdr_mode=self.hdr_mode,
                trt_static_shape=not self.trt_dynamic_shapes,  # if dynamic shapes are enabled, we have to set the static shape to false (default is true in the model
            )
