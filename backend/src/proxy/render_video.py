from src.schemas import RenderSettings, Setting

from src.proxy.backends.interpolate_base import InterpolateBase
from src.proxy.backends.upscale_base import UpscaleBase
from src.proxy.io_buffers.ffmpeg_proxy import ReadBuffer, WriteBuffer
from src.utils.LogConfig import get_logger
from src.proxy.video_info_proxy import VideoInfo

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

    def render(
        self,
        write_buffer: WriteBuffer,
        read_buffer: ReadBuffer,
        interpolate_option: InterpolateBase | None = None,
        upscale_option: UpscaleBase | None = None,
    ):

        frame = read_buffer.get()
        while frame:
            write_buffer.put_frame_in_write_queue(frame)

            frame = read_buffer.get()

        write_buffer.put_frame_in_write_queue(None)
