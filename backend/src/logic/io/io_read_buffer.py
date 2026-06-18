from abc import ABC, abstractmethod

from src.schemas.domain.frame import Frame


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
