from abc import ABC, abstractmethod
from src.schemas.domain import Frame


class InterpolateMethodBase(ABC):
    @abstractmethod
    def process_frame(self, frame: Frame, transition: bool) -> list[Frame]
