from abc import ABC, abstractmethod 
from src.schemas.domain import Frame, InterpolateModel

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
    
    def setup_interpolation(self, interpolate_model: InterpolateModel) -> InterpolateBase:
        if interpolate_model.backend.type == "pytorch":
            