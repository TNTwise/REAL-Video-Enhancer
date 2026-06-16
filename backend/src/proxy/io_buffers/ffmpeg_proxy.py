import asyncio
import shlex
import tempfile
import time
from abc import ABC, abstractmethod

import cv2
import numpy as np

from src.constants import FFMPEG_PATH
from src.proxy.settings import PersistentSettingsProxy
from src.schemas.domain.render import RenderSettings
from src.schemas.domain.video_info import InputVideoInfo, OutputVideoInfo

from ...schemas.domain.frame import Frame
from ...utils.LogConfig import get_logger

logger = get_logger(__name__)


class ReadBuffer(ABC):
    @abstractmethod
    def command(self) -> list[str]:
        pass

    @abstractmethod
    async def read_frames_into_queue(self) -> None:
        pass

    @abstractmethod
    async def get(self) -> Frame | None:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass


class WriteBuffer(ABC):
    @abstractmethod
    def command(self) -> list[str]:
        pass

    @abstractmethod
    async def put_frame_in_write_queue(self, frame: Frame | None) -> None:
        pass

    @abstractmethod
    async def write_out_frames(self) -> None:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass


class FFmpegRead(ReadBuffer):
    def __init__(
        self,
        render_settings: RenderSettings,
        input_video_info: InputVideoInfo,
        output_video_info: OutputVideoInfo,
        settings: PersistentSettingsProxy,
    ):
        self.render_settings = render_settings
        self.video_info = input_video_info
        self.settings = settings

        # TODO: Make yuv mod work
        self._yuv420p_mod = self.video_info.pixel_format == "yuv420p"
        self._yuv420p_mod = False
        if settings.auto_hdr_mode and self.video_info.is_hdr:
            self.input_frame_chunk_size = (
                self.video_info.width * self.video_info.height * 6
            )
        elif self._yuv420p_mod:
            self.input_frame_chunk_size = (
                self.video_info.width * self.video_info.height * 3 // 2
            )
        else:
            self.input_frame_chunk_size = (
                self.video_info.width * self.video_info.height * 3
            )

        self._read_queue: asyncio.Queue[Frame | None] = asyncio.Queue(maxsize=200)
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task | None = None
        self._stderr_file = tempfile.TemporaryFile(
            mode="w+", encoding="utf-8", errors="replace"
        )
        self._closed = False
        self._debug_counter = 0
        if self.settings.auto_hdr_mode and self.video_info.is_hdr:
            self._input_pix_fmt = "rgb48le"
        elif self._yuv420p_mod:
            self._input_pix_fmt = "yuv420p"
        else:
            self._input_pix_fmt = "rgb24"

    def command(self):
        command = [
            f"{FFMPEG_PATH}",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            f"{self.render_settings.input_video_info.input_file}",
            "-f",
            "image2pipe",
            "-pix_fmt",
            "rgb48le"
            if self.settings.auto_hdr_mode and self.video_info.is_hdr
            else (self.video_info.pixel_format if self._yuv420p_mod else "rgb24"),
            "-vcodec",
            "rawvideo",
            "-s",
            f"{self.video_info.width}x{self.video_info.height}",
            "-",
        ]
        logger.info("FFMPEG READ COMMAND: %s", command)
        return command

    async def start(self):
        if self._process is not None:
            return
        command = self.command()
        self._process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=self._stderr_file,
        )
        self._stderr_task = asyncio.create_task(self._read_stderr())

    async def _read_stderr(self):
        if self._process and self._process.stderr:
            await self._process.stderr.read()

    async def read_frames_into_queue(self):
        if self._process is None:
            await self.start()
        assert self._process is not None
        assert self._process.stdout is not None

        try:
            while not self._closed:
                chunk = await self._process.stdout.readexactly(
                    self.input_frame_chunk_size
                )
                if not chunk:
                    break

                if self._yuv420p_mod:
                    pass

                frame = Frame(
                    self.video_info.width,
                    self.video_info.height,
                )
                frame.set_frame_bytes(chunk)
                await self._read_queue.put(frame)
        except asyncio.IncompleteReadError:
            pass
        finally:
            await self._read_queue.put(None)

    async def get(self) -> Frame | None:
        if self._process is None:
            await self.start()
        return await self._read_queue.get()

    async def close(self):
        self._closed = True
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    self._process.kill()
                    await self._process.wait()
                except ProcessLookupError:
                    pass
        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
        self._stderr_file.close()


class FFmpegWrite(WriteBuffer):
    def __init__(
        self,
        render_settings: RenderSettings,
        input_video_info: InputVideoInfo,
        output_video_info: OutputVideoInfo,
        settings: PersistentSettingsProxy,
    ):
        self.render_settings = render_settings
        self.input_video_info = input_video_info
        self.output_video_info = output_video_info
        self.settings = settings

        self.write_queue: asyncio.Queue[Frame | None] = asyncio.Queue(maxsize=200)
        self._process: asyncio.subprocess.Process | None = None
        self._write_task: asyncio.Task | None = None
        self._start_time: float = 0
        self._closed = False
        self._frames_written: int = 0
        self._current_fps: float = 0.0
        self._last_fps_update: float = 0
        self._debug_counter = 0
        self._output_pix_fmt = (
            "rgb48le"
            if self.input_video_info.is_hdr and self.settings.auto_hdr_mode
            else "rgb24"
        )

    def command(self):
        command = [
            FFMPEG_PATH,
            "-loglevel",
            "error",
        ]

        command += [
            "-framerate",
            f"{self.input_video_info.fps}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb48le"
            if self.input_video_info.is_hdr and self.settings.auto_hdr_mode
            else "rgb24",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{self.input_video_info.width}x{self.input_video_info.height}",
            "-i",
            "-",
        ]

        command += [
            "-fflags",
            "+genpts",
            "-i",
            f"{self.render_settings.input_video_info.input_file}",
            "-map",
            "0:v",
            "-map",
            "1:a?",
            "-map",
            "1:s?",
            "-map_metadata:s:v",
            "1:s:v",
            "-metadata:s:v",
            "rotate=0",
        ]

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

        command += [
            "-r",
            f"{self.output_video_info.fps}",
        ]

        if (
            self.settings.use_custom_encoder_command == "True"
            and self.settings.encoder_command.strip()
        ):
            command += shlex.split(self.settings.encoder_command)
        else:
            command += self._encoder_args()

        command += [
            f"{self.output_video_info.output_file}",
        ]

        if self.render_settings.overwrite:
            command.append("-y")

        return command

    def _encoder_args(self) -> list[str]:
        # TODO: Make settings save as actual values, none of this mapping bullshit
        args: list[str] = []

        encoder_map = {
            "libx264": "libx264",
            "libx265": "libx265",
            "vp9": "libvpx-vp9",
            "av1": "libaom-av1",
            "prores": "prores_ks",
            "ffv1": "ffv1",
            "utvideo": "utvideo",
            "x264_nvenc": "h264_nvenc",
            "x265_nvenc": "hevc_nvenc",
            "av1_nvenc (40 series and up)": "av1_nvenc",
        }

        quality = self.settings.video_quality

        crf_values = {
            "Lossless": "0",
            "Ultra": "10",
            "Very_High": "14",
            "High": "18",
            "Medium": "23",
            "Low": "28",
        }
        crf = crf_values.get(quality, "18")

        encoder = self.settings.encoder
        ffmpeg_encoder = encoder_map.get(encoder, encoder)
        args += ["-c:v", ffmpeg_encoder]

        if encoder in ("libx264", "libx265"):
            if quality == "Lossless":
                if encoder == "libx264":
                    args += ["-qp", "0"]
                else:
                    args += ["-x265-params", "lossless=1"]
            else:
                args += ["-crf", crf]
            preset_map = {
                "placebo": "placebo",
                "slow": "slow",
                "medium": "medium",
                "fast": "fast",
                "fastest": "ultrafast",
            }
            args += [
                "-preset",
                preset_map.get(self.settings.video_encoder_speed, "medium"),
            ]

        elif encoder == "vp9":
            if quality == "Lossless":
                args += ["-lossless", "1"]
            else:
                args += ["-crf", crf]
            speed_map = {
                "placebo": "0",
                "slow": "0",
                "medium": "1",
                "fast": "2",
                "fastest": "3",
            }
            args += ["-cpu-used", speed_map.get(self.settings.video_encoder_speed, "1")]
            args += [
                "-deadline",
                "best"
                if self.settings.video_encoder_speed in ("placebo", "slow")
                else "good",
            ]

        elif encoder in ("x264_nvenc", "x265_nvenc", "av1_nvenc (40 series and up)"):
            if quality == "Lossless":
                args += ["-qp", "0"]
            else:
                args += ["-cq", crf]
            nvenc_presets = {
                "placebo": "p7",
                "slow": "p7",
                "medium": "p5",
                "fast": "p3",
                "fastest": "p1",
            }
            args += [
                "-preset",
                nvenc_presets.get(self.settings.video_encoder_speed, "p5"),
            ]

        elif encoder == "prores":
            prores_map = {
                "Lossless": "5",
                "Ultra": "4",
                "Very_High": "3",
                "High": "2",
                "Medium": "1",
                "Low": "0",
            }
            args += ["-profile:v", prores_map.get(quality, "2")]

        elif encoder == "av1":
            if quality == "Lossless":
                args += ["-lossless", "1"]
            else:
                args += ["-crf", crf]

        elif encoder in ("ffv1", "utvideo"):
            pass

        # Output pixel format
        args += ["-pix_fmt", self.settings.video_pixel_format]

        # Audio
        audio_enc = self.settings.audio_encoder
        if audio_enc == "copy_audio":
            args += ["-c:a", "copy"]
        else:
            args += ["-c:a", audio_enc]
        args += ["-b:a", self.settings.audio_bitrate]

        # Subtitles
        sub_enc = self.settings.subtitle_encoder
        if sub_enc == "copy_subtitle":
            args += ["-c:s", "copy"]
        else:
            args += ["-c:s", sub_enc]

        # Container
        container_map = {
            "mkv": "matroska",
            "mp4": "mp4",
            "mov": "mov",
            "webm": "webm",
            "avi": "avi",
        }
        args += [
            "-f",
            container_map.get(
                self.settings.video_container, self.settings.video_container
            ),
        ]

        return args

    async def start(self):
        if self._process is not None:
            return
        command = self.command()

        logger.info("FFMPEG WRITE COMMAND: %s", command)
        import sys

        self._process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=sys.stdout,
            stderr=asyncio.subprocess.STDOUT,
        )

    async def put_frame_in_write_queue(self, frame: Frame | None) -> None:
        if self._process is None:
            await self.start()
        await self.write_queue.put(frame)

    async def _write_loop(self):
        if self._process is None:
            await self.start()
        assert self._process is not None
        assert self._process.stdin is not None

        try:
            while not self._closed:
                frame = await self.write_queue.get()
                if frame is None:
                    break

                frame_bytes = frame.get_frame_bytes()
                self._process.stdin.write(frame_bytes)
                await self._process.stdin.drain()

                self._frames_written += 1
                now = time.time()
                if now - self._last_fps_update >= 1.0:
                    elapsed = now - self._start_time
                    if elapsed > 0:
                        self._current_fps = self._frames_written / elapsed
                    self._last_fps_update = now
        except (BrokenPipeError, ConnectionResetError):
            logger.warning("FFmpeg write pipe broken")
        finally:
            if self._process and self._process.stdin:
                self._process.stdin.close()

    async def write_out_frames(self):
        logger.info("Rendering")
        self._start_time = time.time()

        self._write_task = asyncio.create_task(self._write_loop())
        await self._write_task

        if self._process:
            try:
                await asyncio.wait_for(self._process.wait(), timeout=30.0)
                exit_code = self._process.returncode
            except asyncio.TimeoutError:
                logger.warning("FFmpeg write process timeout, killing")
                self._process.kill()
                await self._process.wait()
                exit_code = -1

            if exit_code != 0:
                logger.info("FFmpeg exited with code %s", exit_code)
            else:
                render_time = time.time() - self._start_time
                logger.info("Time to complete render: %s", round(render_time, 2))

    def get_current_fps(self) -> float:
        return self._current_fps

    def get_frames_written(self) -> int:
        return self._frames_written

    async def close(self):
        self._closed = True
        if self._process:
            try:
                if self._process.stdin:
                    self._process.stdin.close()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    self._process.kill()
                    await self._process.wait()
                except ProcessLookupError:
                    pass
