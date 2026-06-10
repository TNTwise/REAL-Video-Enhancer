from abc import ABC, abstractmethod 
from src.schemas.domain import Frame

class InterpolateBase(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(self, img1: Frame, transition=False):
        raise NotImplementedError("Subclasses must implement this method")