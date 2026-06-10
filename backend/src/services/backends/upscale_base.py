from abc import abstractmethod, ABC

from backend.src.schemas.domain.frame import Frame

class UpscaleBase(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def __call__(self, img: Frame):
        raise NotImplementedError("Subclasses must implement this method")