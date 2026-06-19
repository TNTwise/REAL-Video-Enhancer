import json
from abc import ABC, abstractmethod

from src.logic.proxy import PersistentSettingsProxy
from src.schemas.domain.frame import Frame
from src.schemas.domain.render import RenderSettings
from src.schemas.domain.video_info import InputVideoInfo, OutputVideoInfo
from src.utils.LogConfig import get_logger

logger = get_logger(__name__)


class WriteBuffer(ABC):
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
        logger.info(
            "Input Video Info:\n%s",
            json.dumps(input_video_info.model_dump(), indent=2, default=str),
        )
        logger.info(
            "Output Video Info:\n%s",
            json.dumps(output_video_info.model_dump(), indent=2, default=str),
        )

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
