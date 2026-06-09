from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal, Annotated, Union


class InterpolateModelClientInput(BaseModel):
    type: Literal["interpolate"] = "interpolate"
    id: str


class UpscaleModelClientInput(BaseModel):
    type: Literal["upscale"] = "upscale"
    id: str


class EnhancementModelClientInput(BaseModel):
    type: Literal["enhancement"] = "enhancement"
    id: str


ModelInputVariant = Annotated[
    Union[
        InterpolateModelClientInput,
        UpscaleModelClientInput,
        EnhancementModelClientInput,
    ],
    Field(discriminator="type"),
]
