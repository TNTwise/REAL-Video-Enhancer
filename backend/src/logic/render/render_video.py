import asyncio

from src.logic.io import ReadBuffer, WriteBuffer
from src.logic.proxy import PersistentSettingsProxy
from src.logic.render.methods.interpolate.interpolate_method_base import (
    InterpolateMethodBase,
)
from src.schemas.domain import RenderSettings
from src.utils.LogConfig import get_logger

logger = get_logger(__name__)


class RenderProxy:
    async def render(
        self,
        write_buffer: WriteBuffer,
        read_buffer: ReadBuffer,
        render_settings: RenderSettings,
        persistent_settings: PersistentSettingsProxy,
        interpolate_method: InterpolateMethodBase | None,
    ):
        async def reader_task():
            await read_buffer.read_frames_into_queue()

        async def processor_task():
            while frame := await read_buffer.get():
                if frame is None:
                    break
                if interpolate_method:
                    interpolated_frames = interpolate_method.process_frame(frame, False)

                    for interpolated_frame in interpolated_frames:
                        await write_buffer.put_frame_in_write_queue(interpolated_frame)

                await write_buffer.put_frame_in_write_queue(frame)
            await write_buffer.put_frame_in_write_queue(None)

        async def writer_task():
            await write_buffer.write_out_frames()

        await asyncio.gather(
            reader_task(),
            processor_task(),
            writer_task(),
        )
