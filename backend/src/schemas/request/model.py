from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class InterpolateModelClientInput(BaseModel):
    type: Literal["interpolate"] = "interpolate"
    id: str
    backed_type: str
    interpolate_factor: int


class UpscaleModelClientInput(BaseModel):
    type: Literal["upscale"] = "upscale"
    id: str
    backed_type: str


class EnhancementModelClientInput(BaseModel):
    type: Literal["enhancement"] = "enhancement"
    id: str
    backed_type: str


ModelInputVariant = Annotated[
    Union[
        InterpolateModelClientInput,
        UpscaleModelClientInput,
        EnhancementModelClientInput,
    ],
    Field(discriminator="type"),
]
