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
        logger.info("Using backend: %s", self.render_settings.backend)
        if render_settings.upscale_mode:
            self.setupUpscale()
            self.upscaleOption.hotUnload()  # unload model to free up memory for trt enging building
            logger.info("Using Upscaling Model: %s", self.render_settings.upscale_model)

        if render_settings.interpolate_model:
            self.setupInterpolate()
            logger.info("Using Interpolation Model: %s", self.interpolateModel)

        if render_settings.upscale_model:
            self.upscaleOption.hotReload()

        for extraRestoration in self.render_settings.extra_restoration_models:
            extraRestoration.hotReload()


    def render(self,
               interpolate_option: ):
        frame = self.read_buffer.get()
        while frame:
            for extraRestoration in self.render_settings.extra_restoration_models:
                frame = extraRestoration(frame)

            if self.render_settings.interpolate_model:
                interpolated_frames = self.interpolateOption(
                    img1=frame,
                    transition=False,
                )

                for interpolated_frame in interpolated_frames:
                    if self.render_settings.upscale_model:
                        interpolated_frame = self.upscaleOption(interpolated_frame)

                    self.write_buffer.put_frame_in_write_queue(interpolated_frame)

            if self.render_settings.upscale_model:
                frame = self.upscaleOption(frame)

            if self.override_upscale_scale:
                frame = frame.resize_frame_optimal(
                    self.width * self.override_upscale_scale,
                    self.height * self.override_upscale_scale,
                )


            self.write_buffer.put_frame_in_write_queue(frame)

            # grab new frame
            frame = self.read_buffer.get()

        self.informationHandler.stopWriting()
        self.write_buffer.writeQueue.put(None)
    
