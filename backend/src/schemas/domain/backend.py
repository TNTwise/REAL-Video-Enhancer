from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


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
