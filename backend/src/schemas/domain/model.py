from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal, Annotated, Union


class InterpolateModel(BaseModel):
    type: Literal["interpolate"] = "interpolate"
    variant: str
    url: str


class UpscaleModel(BaseModel):
    type: Literal["upscale"] = "upscale"
    variant: str
    scale: int
    url: str


class EnhancementModel(BaseModel):
    type: Literal["enhancement"] = "enhancement"
    variant: str
    url: str


ModelVariant = Annotated[
    Union[InterpolateModel, UpscaleModel, EnhancementModel], Field(discriminator="type")
]
