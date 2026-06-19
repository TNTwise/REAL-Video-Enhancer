from __future__ import annotations

from typing import Annotated, Literal, Union

import numpy
from pydantic import BaseModel, Field
from src.logic.handlers.backend import TorchHandler


class TorchPrecision(BaseModel):
    """precision id: (float16, float32, etc)"""

    type: Literal["torch"] = "torch"
    precision_id: str

    @property
    def dtype(self):
        if not TorchHandler().is_available():
            raise RuntimeError("PyTorch is not available")

        return getattr(TorchHandler().get_torch(), self.precision_id)


class NumpyPrecision(BaseModel):
    """precision id: (float16, float32, etc)"""

    type: Literal["numpy"] = "numpy"
    precision_id: str

    @property
    def dtype(self):
        return getattr(numpy, self.precision_id)


Precision = Annotated[
    Union[NumpyPrecision, TorchPrecision], Field(discriminator="type")
]
