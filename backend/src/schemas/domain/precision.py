from pydantic import BaseModel
from src.handlers import TorchHandler
import numpy
from pydantic import BaseModel, Field
from typing import Annotated, Union
    


class TorchPrecision(BaseModel):
    """precision id: (float16, float32, etc)"""

    precision_id: str

    @property
    def dtype(self):
        if not TorchHandler().is_available():
            raise RuntimeError("PyTorch is not available")

        return getattr(TorchHandler().get_torch(), self.precision_id)

class NumpyPrecision(BaseModel):
    """precision id: (float16, float32, etc)"""

    precision_id: str

    @property
    def dtype(self):
        return getattr(numpy, self.precision_id)


Precision = Annotated[
    Union[NumpyPrecision, TorchPrecision], Field(discriminator="type")
]
