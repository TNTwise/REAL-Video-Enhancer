from abc import ABC, abstractmethod

from src.schemas.domain import Frame, UpscaleModel


class UpscaleBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, img: Frame):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def hot_reload(self):
        """Reload the upscaling model."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def setup_upscale(self, upscale_model: UpscaleModel):
        """Set up the upscaling model."""
        raise NotImplementedError("Subclasses must implement this method")
