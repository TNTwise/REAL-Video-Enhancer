from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal, Annotated, Union


class InterpolateModel(BaseModel):
    id: str
    type: Literal["interpolate"] = "interpolate"
    variant: str
    file_path: str
    description: str | None = None
    url: str | None = None


class UpscaleModel(BaseModel):
    id: str
    type: Literal["upscale"] = "upscale"
    variant: str
    scale: int
    file_path: str
    description: str | None = None
    url: str | None = None


class EnhancementModel(BaseModel):
    type: Literal["enhancement"] = "enhancement"
    id: str
    variant: str
    file_path: str
    description: str | None = None
    url: str | None = None



ModelVariant = Annotated[
    Union[InterpolateModel, UpscaleModel, EnhancementModel], Field(discriminator="type")
]
