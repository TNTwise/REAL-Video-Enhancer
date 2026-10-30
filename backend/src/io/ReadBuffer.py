from abc import ABC, abstractmethod

from ..utils.Frame import Frame


class ReadBuffer(ABC):
    @abstractmethod
    def command(self) -> list[str]:
        pass

    @abstractmethod
    def read_frame(self) -> bytes | None:
        pass

    @abstractmethod
    def read_frames_into_queue(self):
        pass

    @abstractmethod
    def get(self) -> Frame:
        pass

    @abstractmethod
    def close(self):
        pass
