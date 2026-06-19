from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field
from src.schemas.domain.backend import Backend


class InterpolateModelRepo(BaseModel):
    id: str
    type: Literal["interpolate"] = "interpolate"
    variant: str
    interpolate_factor: int
    file_path: str
    description: str | None = None
    url: str | None = None
    precision: str = "float32"
    backend: Backend


class UpscaleModelRepo(BaseModel):
    id: str
    type: Literal["upscale"] = "upscale"
    variant: str
    scale: int
    file_path: str
    description: str | None = None
    url: str | None = None
    precision: str = "float32"
    backend: Backend


class EnhancementModelRepo(BaseModel):
    type: Literal["enhancement"] = "enhancement"
    id: str
    variant: str
    file_path: str
    description: str | None = None
    url: str | None = None
    precision: str = "float32"
    backend: Backend


ModelRepoVariant = Annotated[
    Union[InterpolateModelRepo, UpscaleModelRepo, EnhancementModelRepo],
    Field(discriminator="type"),
]
