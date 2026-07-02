import asyncio
import time

from src.logic.io import ReadBuffer, WriteBuffer
from src.logic.proxy import PersistentSettingsProxy
from src.logic.render.methods.interpolate.interpolate_method_base import (
    InterpolateMethodBase,
)
from src.logic.render.methods.upscale.upscale_method_base import UpscaleMethodBase
from src.schemas.domain import Frame, RenderSettings
from src.utils.LogConfig import get_logger

logger = get_logger(__name__)


def _process_frame(
    frame: Frame,
    interpolate_method: InterpolateMethodBase | None,
    upscale_method: UpscaleMethodBase | None,
) -> list[Frame]:
    result: list[Frame] = []
    if interpolate_method:
        for f in interpolate_method.process_frame(frame, False):
            if upscale_method:
                f = upscale_method.process_frame(f)
            result.append(f)
    if upscale_method:
        frame = upscale_method.process_frame(frame)
    result.append(frame)
    return result


class RenderProxy:
    def __init__(self):
        self.current_frame_bytes: bytes | None = None
        self.current_fps: float = 0.0
        self.current_frame_number: int = 0
        self.frame_width: int = 0
        self.frame_height: int = 0

    async def render(
        self,
        write_buffer: WriteBuffer,
        read_buffer: ReadBuffer,
        render_settings: RenderSettings,
        persistent_settings: PersistentSettingsProxy,
        interpolate_method: InterpolateMethodBase | None,
        upscale_method: UpscaleMethodBase | None,
    ):
        async def reader_task():
            await read_buffer.read_frames_into_queue()

        async def processor_task():
            frame_count = 0
            start_time = time.time()

            while frame := await read_buffer.get():
                if frame is None:
                    break

                if frame_count == 0:
                    self.frame_width = frame.width
                    self.frame_height = frame.height

                processed = await asyncio.to_thread(
                    _process_frame, frame, interpolate_method, upscale_method
                )

                for processed_frame in processed:
                    await write_buffer.put_frame_in_write_queue(processed_frame)
                    frame_count += 1
                    self.current_frame_bytes = processed_frame.get_frame_bytes()
                    self.current_frame_number = frame_count
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        self.current_fps = frame_count / elapsed

            await write_buffer.put_frame_in_write_queue(None)

        async def writer_task():
            await write_buffer.write_out_frames()

        await asyncio.gather(
            reader_task(),
            processor_task(),
            writer_task(),
        )
