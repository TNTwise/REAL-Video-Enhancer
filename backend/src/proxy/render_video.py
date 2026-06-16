import asyncio

from src.proxy.backends.interpolate_base import InterpolateBase
from src.proxy.backends.upscale_base import UpscaleBase
from src.proxy.io_buffers.ffmpeg_proxy import ReadBuffer, WriteBuffer
from src.schemas.domain import RenderSettings
from src.schemas.request import Setting
from src.utils.LogConfig import get_logger

logger = get_logger(__name__)


class RenderProxy:
    def __init__(
        self,
        render_settings: RenderSettings,
        settings: Setting,
    ):
        self.render_settings = render_settings
        self.settings = settings

    async def render(
        self,
        write_buffer: WriteBuffer,
        read_buffer: ReadBuffer,
        interpolate_option: InterpolateBase | None = None,
        upscale_option: UpscaleBase | None = None,
    ):
        async def reader_task():
            await read_buffer.read_frames_into_queue()

        async def processor_task():
            while True:
                frame = await read_buffer.get()
                if frame is None:
                    break
                processed_frame = await self._process_frame(
                    frame, interpolate_option, upscale_option
                )
                await write_buffer.put_frame_in_write_queue(processed_frame)
            await write_buffer.put_frame_in_write_queue(None)

        async def writer_task():
            await write_buffer.write_out_frames()

        await asyncio.gather(
            reader_task(),
            processor_task(),
            writer_task(),
        )

    async def _process_frame(
        self,
        frame,
        interpolate_option: InterpolateBase | None,
        upscale_option: UpscaleBase | None,
    ):
        if interpolate_option is not None:
            frame = await interpolate_option.process_frame(frame)
        if upscale_option is not None:
            frame = await upscale_option.process_frame(frame)
        return frame
