import os
import pathlib
import queue
import subprocess
import time
import shlex
from abc import ABC, abstractmethod

from src.constants import FFMPEG_PATH
from src.schemas.domain.render import RenderSettings
from backend.src.proxy.settings import Settings
from backend.src.proxy.video_info_proxy import OpenCVInfo
from src.utils import BorderDetect
import cv2
import numpy as np
import tempfile

from ...schemas.domain.frame import Frame
from ...utils.LogConfig import get_logger
from ...utils.Util import (
    subprocess_popen_without_terminal,
)

logger = get_logger(__name__)


class ReadBuffer(ABC):
    @abstractmethod
    def command(self) -> list[str]:
        """Build the FFmpeg command for reading frames from the source video."""
        pass

    @abstractmethod
    def read_frame(self) -> bytes | None:
        """Read a single raw frame from the FFmpeg stdout pipe. Returns None on EOF."""
        pass

    @abstractmethod
    def read_frames_into_queue(self) -> None:
        """Read all frames into the internal queue, sentinel None at the end."""
        pass

    @abstractmethod
    def get(self) -> Frame:
        """Get the next processed Frame from the internal queue."""
        pass


class WriteBuffer(ABC):
    @abstractmethod
    def command(self) -> list[str]:
        """Build the FFmpeg command for writing encoded output video."""
        pass

    @abstractmethod
    def get_num_frames_rendered(self) -> int:
        """Return the current count of frames that have been rendered."""
        pass

    @abstractmethod
    def put_frame_in_write_queue(self, frame: Frame) -> None:
        """Enqueue a processed Frame for writing to the FFmpeg stdin pipe."""
        pass

    @abstractmethod
    def write_out_frames(self) -> None:
        """Drain the write queue and feed raw frames into the FFmpeg process."""
        pass

    @abstractmethod
    def onErroredExit(self) -> None:
        """Handle an error exit from the FFmpeg write process."""
        pass


class FFmpegRead(ReadBuffer):
    def __init__(
        self,
        render_settings: RenderSettings,
        video_info: OpenCVInfo,
        settings: Settings,
        border_detect: BorderDetect,
    ):
        self.render_settings = render_settings
        self.video_info = video_info
        self.settings = settings
        self.border_detect = border_detect

        self._yuv420p_mod = video_info.pixel_format == "yuv420p"
        if render_settings.hdr_mode:
            self.input_frame_chunk_size = video_info.width * video_info.height * 6
        elif self._yuv420p_mod:
            self.input_frame_chunk_size = video_info.width * video_info.height * 3 // 2
        else:
            self.input_frame_chunk_size = video_info.width * video_info.height * 3
        command = self.command()
        logger.info("FFMPEG READ COMMAND: %s", command)

        self.stderr_file = tempfile.TemporaryFile(
            mode="w+", encoding="utf-8", errors="replace"
        )

        self._read_process = subprocess_popen_without_terminal(
            command,
            stdout=subprocess.PIPE,
            stderr=self.stderr_file,
        )
        self._read_queue = queue.Queue(maxsize=25)

    def command(self):
        # will have to figure out a cleaner solution to this later
        # border_width, border_height, border_x, border_y = self.border_detect.get_borders()
        # filter_string = f"crop=min({self.video_info.width}\\,max(1\\,iw-{border_x})):min({self.video_info.height}\\,max(1\\,ih-{border_y})):{border_x}:{border_y},scale=if(gt(sar\\,0)\\,trunc(iw*max(sar\\,0)/2)*2\\,iw):ih,setsar=1"  # fix dar != sar

        command = [
            f"{FFMPEG_PATH}",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            f"{self.render_settings.video_path}",
            #    "-vf",
            #    filter_string,
            "-f",
            "image2pipe",
            "-pix_fmt",
            "rgb48le"
            if self.render_settings.hdr_mode
            else (self.video_info.pixel_format if self._yuv420p_mod else "rgb24"),
            "-vcodec",
            "rawvideo",
            "-s",
            f"{self.video_info.width}x{self.video_info.height}",
            "-",
        ]

        logger.info("FFMPEG READ COMMAND: %s", command)
        return command

    def read_frame(self):
        chunk = self._read_process.stdout.read(self.input_frame_chunk_size)
        if len(chunk) < self.input_frame_chunk_size:
            return None

        if self._yuv420p_mod:
            # Convert raw YUV420p data to RGB
            # The data is Y plane, then U plane, then V plane, concatenated.
            # cv2.COLOR_YUV420P2RGB expects a single channel image of shape (height * 3 // 2, width)
            np_frame = np.frombuffer(chunk, dtype=np.uint8)
            # Ensure height is an integer for reshape, Python 3 // operator already does this.
            yuv_image_height = self.height * 3 // 2
            yuv_image = np_frame.reshape((yuv_image_height, self.width))
            rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB_I420)
            # cv2.imwrite("temp_rgb_image.png", rgb_image)  # Debugging line, can be removed
            chunk = rgb_image.tobytes()

        return chunk

    def read_frames_into_queue(self):
        while True:
            chunk = self.read_frame()
            if chunk is None:
                break
            frame = Frame(
                self.backend,
                self.width,
                self.height,
                self.device,
                self.gpu_id,
                self.hdr_mode,
                self.dtype,
            )
            frame.set_frame_bytes(chunk)
            self._read_queue.put(frame)
        self._read_queue.put(None)

    def get(self) -> Frame:
        return self._read_queue.get()

    def __del__(self):
        self._read_process.stdout.close()
        if self._read_process.returncode != 0:
            self.stderr_file.seek(0)
            stderr_output = self.stderr_file.read()
            logger.info("FFmpeg Read Process stderr:\n%s", stderr_output)
        self._read_process.terminate()
        self.stderr_file.close()


class FFmpegWrite(WriteBuffer):
    def __init__(
        self,
        render_settings: RenderSettings,
        video_info: OpenCVInfo,
        settings: Settings,
    ):
        self.render_settings = render_settings
        self.video_info = video_info
        self.settings = settings
        # inputFPS reflects the actual rate of frames the model produces (using ceil)
        # For integer factors, inputFPS == outputFPS (no frame dropping).
        # For decimal factors (e.g. 2.5x), inputFPS > outputFPS and FFmpeg
        # drops the excess frames to achieve the correct target FPS.
        try:
            command = self.command()
            logger.info("FFMPEG WRITE COMMAND: %s", command)
            self.writeProcess = subprocess_popen_without_terminal(
                command,
                stdin=subprocess.PIPE,
                stderr=self.ffmpeg_log,
                stdout=subprocess.PIPE if self.mpv_output else self.ffmpeg_log,
                text=True,
                universal_newlines=True,
            )
        except Exception as e:
            logger.info(e.__str__())
            logger.exception("Exception while starting FFmpeg write process")
            self.onErroredExit()

    def command(self):
            # maybe i can split this so i can just use ffmpeg normally like with vspipe
            command = [
                f"{self.ffmpeg_path}",
                "-loglevel",
                "error",
            ]

            if self.render_settings.upscale_model:
                output_width = self.video_info.width * self.render_settings.upscale_model.scale
                output_height = self.video_info.height * self.render_settings.upscale_model.scale
            else:
                output_width = self.video_info.width
                output_height = self.video_info.height

            if self.render_settings.interpolate_model:
                output_fps = self.render_settings.interpolate_model.fps
            else:
                output_fps = self.video_info.fps

            command += [
                "-framerate",
                f"{self.video_info.fps}",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb48le" if self.hdr_mode else "rgb24",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{output_width}x{output_height}",
                "-i",
                "-",
            ]

            command += [
                    # Input 1: original file for audio/subtitles.
                    # Put timestamp hygiene flags *before* the input they apply to.
                    "-fflags",
                    "+genpts",
                    "-i",
                    f"{self.render_settings.video_path}",
                    "-map",
                    "0:v",  # Map video stream from input 0
                    "-map",
                    "1:a?",
                    "-map",
                    "1:s?",
                    "-map_metadata:s:v",
                    "1:s:v",  # Copy video stream metadata from input 1 (the original file) to the video output
                    "-metadata:s:v",
                    "rotate=0",  # Ensure custom rotation is stripped as the output is physically rotated
                ]

                # Output timestamp/interleave hygiene.
            command += [
                    "-avoid_negative_ts",
                    "make_zero",
                    "-max_interleave_delta",
                    "0",
                    "-muxpreload",
                    "0",
                    "-muxdelay",
                    "0",
                ]

            # Output frame rate must come after all -i inputs
            # so FFmpeg treats it as an output option, not an input option.
            command += [
                "-r",
                f"{output_fps}",
            ]

            command += [
                f"{self.render_settings.output_path}",
            ]

            if self.render_settings.overwrite:
                command.append("-y")


            return command

    def get_num_frames_rendered(self):
        return self.framesRendered

    def put_frame_in_write_queue(self, frame):
        self.writeQueue.put(frame)

    def write_out_frames(self):
        logger.info("Rendering")
        self.startTime = time.time()

        exit_code: int = 0
        try:
            while True:
                frame = self.writeQueue.get()
                if frame is None:
                    break

                self.writeProcess.stdin.buffer.write(frame)

            self.writeProcess.stdin.close()
            self.writeProcess.wait()
            exit_code = self.writeProcess.returncode

        except Exception:
            logger.exception("Exception while writing frames")
            self.onErroredExit()

        if exit_code != 0:
            logger.info("Exception while writing frames")
            logger.info("FFmpeg exited with code %s", exit_code)
        else:
            renderTime = time.time() - self.startTime
            logger.info("Time to complete render: %s", round(renderTime, 2))

    def onErroredExit(self):
        logger.info("FFmpeg failed to render the video.")
        try:
            with pathlib.Path(self.ffmpeg_log_file).open("r") as f:
                logger.info("FULL FFMPEG LOG:")
                for line in f:
                    logger.info("%s", line.rstrip("\n"))

            with pathlib.Path(self.ffmpeg_log_file).open("r") as f:
                for line in f:
                    if f"[{self.outputFileExtension}" in line:
                        logger.info("%s", line.rstrip("\n"))

            if self.video_encoder.getPresetTag() == "x264_vulkan":
                logger.info("Vulkan encode failed, try restarting the render.")
                logger.info(
                    "Make sure you have the latest drivers installed and your GPU supports vulkan encoding."
                )
        except Exception:
            logger.exception("Failed to read FFmpeg log file")

        logger.info("Time to complete render: Nan")
        time.sleep(1)
        os._exit(1)

    def __del__(self):
        self.ffmpeg_log.close()


class MPVOutput:
    def __init__(
        self, FFMpegWrite: FFmpegWrite, width, height, fps, outputFrameChunkSize
    ):
        self.proc = None
        self.startTime = time.time()
        self.FFMPegWrite = FFMpegWrite
        self.outputFrameChunkSize = outputFrameChunkSize
        self.width = width
        self.height = height
        self.fps = fps

    def command(self):
        command = [
            "mpv",
            f"--audio-file={self.FFMPegWrite.inputFile}",
            "--no-config",
            "--cache=yes",
            "--cache-secs=5",  # Cache 30 seconds of video
            "--demuxer-max-bytes=500Mib",  # Increase max bytes
            "--demuxer-readahead-secs=5",  # Read ahead 30 seconds
            "--demuxer-seekable-cache=yes",  # Enable seekable cache
            "--stream-buffer-size=500MiB",  # Increase buffer size
            "--hr-seek-framedrop=no",  # Prevent frame dropping during seeks
            "-",
        ]
        return command

    def write_out_frames(self):
        with pathlib.Path("mpv_log.txt").open("w") as f:
            while not self.FFMPegWrite.writeProcess:
                time.sleep(1)
            self.proc = subprocess_popen_without_terminal(
                self.command(),
                stdin=self.FFMPegWrite.writeProcess.stdout,
                stderr=f,
                stdout=f,
            )
            self.FFMPegWrite.writeProcess.stdout.close()
            self.proc.wait()
            self.stop()
            os._exit(0)  # force exit

    def stop(self):
        """
        Stop mpv by closing stdin.
        """
        if self.proc:
            self.proc.terminate()
            self.proc.wait()
