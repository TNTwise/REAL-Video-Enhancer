from __future__ import annotations

from src.schemas.domain.precision import Precision
from pydantic import BaseModel, Field
from typing import Literal, Annotated, Union
from src.schemas.domain.backend import Backend

class InterpolateModel(BaseModel):
    id: str
    type: Literal["interpolate"] = "interpolate"
    variant: str
    interpolate_factor: int
    file_path: str
    description: str | None = None
    url: str | None = None
    precision: Precision
    backend: Backend


class UpscaleModel(BaseModel):
    id: str
    type: Literal["upscale"] = "upscale"
    variant: str
    scale: int
    file_path: str
    precision: Precision
    description: str | None = None
    url: str | None = None
    backend: Backend


class EnhancementModel(BaseModel):
    type: Literal["enhancement"] = "enhancement"
    id: str
    variant: str
    file_path: str
    precision: Precision
    description: str | None = None
    url: str | None = None
    backend: Backend


ModelVariant = Annotated[
    Union[InterpolateModel, UpscaleModel, EnhancementModel], Field(discriminator="type")
]
