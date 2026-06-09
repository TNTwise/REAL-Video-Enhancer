from pydantic import BaseModel
from src.handlers import TorchHandler
import numpy

class Precision(BaseModel):
    """precision id: (float16, float32, etc)"""
    precision_id: str

    @property
    def torch_dtype(self):
        if not TorchHandler().is_available():
            raise RuntimeError("PyTorch is not available")
        
        return getattr(TorchHandler().get_torch(), self.precision_id)

    @property
    def numpy_dtype(self):
        return getattr(numpy, self.precision_id)