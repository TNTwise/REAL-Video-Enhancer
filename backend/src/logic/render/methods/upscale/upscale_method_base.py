from abc import ABC, abstractmethod

from src.schemas.domain import Frame


class UpscaleMethodBase(ABC):
    @abstractmethod
    def process_frame(self, frame: Frame) -> Frame: ...
