from src.schemas.domain import Frame, InterpolateModel
from abc import abstractmethod, ABC


class InterpolateBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, img1: Frame, transition=False):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def hot_reload(self):
        """Reload the interpolation model."""
        raise NotImplementedError("Subclasses must implement this method")

    def process_frame(self, frame: Frame):
        pass

    def setup_interpolation(
        self, interpolate_model: InterpolateModel
    ) -> InterpolateBase:
        pass

