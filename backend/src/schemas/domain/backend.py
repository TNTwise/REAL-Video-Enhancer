from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal, Union, Annotated


class NCNNBackend(BaseModel):
    type: Literal["ncnn"] = "ncnn"
    version: str
    installed: bool


class PyTorchBackend(BaseModel):
    type: Literal["pytorch"] = "pytorch"
    version: str
    installed: bool
    accelerator: str


class TensorRTBackend(BaseModel):
    type: Literal["tensorrt"] = "tensorrt"
    version: str
    installed: bool


Backend = Annotated[
    Union[NCNNBackend, PyTorchBackend, TensorRTBackend], Field(discriminator="type")
]
