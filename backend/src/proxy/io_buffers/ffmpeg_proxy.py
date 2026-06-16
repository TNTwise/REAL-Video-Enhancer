import asyncio
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

        self._yuv420p_mod = self.video_info.pixel_format == "yuv420p"
        if render_settings.hdr_mode:
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
            "rgb48le" if self.render_settings.hdr_mode else "rgb24",
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

        command += [
            f"{self.output_video_info.output_file}",
        ]

        if self.render_settings.overwrite:
            command.append("-y")

        return command

    async def start(self):
        if self._process is not None:
            return
        command = self.command()
        logger.info("FFMPEG WRITE COMMAND: %s", command)
        self._process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
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
