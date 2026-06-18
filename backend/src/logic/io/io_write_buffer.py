from abc import ABC, abstractmethod

from src.schemas.domain.frame import Frame


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
