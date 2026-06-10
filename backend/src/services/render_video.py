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

from backend.src.proxy.backends.interpolate_base import InterpolateBase
from backend.src.proxy.backends.upscale_base import UpscaleBase
from backend.src.proxy.io_buffers.ffmpeg_proxy import ReadBuffer, WriteBuffer
from src.InformationWriteOut import InformationWriteOut
from src.utils.LogConfig import get_logger
from src.utils.SceneDetect import SceneDetect
from src.utils.BorderDetect import BorderDetect
from backend.src.proxy.video_info_proxy import VideoInfo

logger = get_logger(__name__)

class Render:
    def __init__(
        self,
        render_settings: RenderSettings,
        settings: Setting,
        video_info: VideoInfo,
    ):
        self.render_settings = render_settings
        self.settings = settings
        self.video_info = video_info
        if render_settings.upscale_model:
            self.setupUpscale()
            self.upscaleOption.hotUnload()  # unload model to free up memory for trt enging building
            logger.info("Using Upscaling Model: %s", self.render_settings.upscale_model)

        if render_settings.interpolate_model:
            self.setupInterpolate()
            logger.info("Using Interpolation Model: %s", self.interpolateModel)

        if render_settings.upscale_model:
            self.upscaleOption.hotReload()

        for extraRestoration in render_settings.extra_restoration_models:
            extraRestoration.hotReload()

    def render(self,
               interpolate_option: InterpolateBase,
               upscale_option: UpscaleBase,
               read_buffer: ReadBuffer,
               write_buffer: WriteBuffer):

        frame = read_buffer.get()
        while frame:
            for extraRestoration in self.render_settings.extra_restoration_models:
                frame = extraRestoration(frame)

            if self.render_settings.interpolate_model:
                interpolated_frames = interpolate_option(
                    img1=frame,
                    transition=False,
                )

                for interpolated_frame in interpolated_frames:
                    if self.render_settings.upscale_model:
                        interpolated_frame = upscale_option(interpolated_frame)

                    write_buffer.put_frame_in_write_queue(interpolated_frame)

            if self.render_settings.upscale_model:
                frame = upscale_option(frame)

            write_buffer.put_frame_in_write_queue(frame)

            frame = read_buffer.get()

        write_buffer.writeQueue.put(None)
    
