from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal, Annotated, Union


class InterpolateModelClientInput(BaseModel):
    type: Literal["interpolate"] = "interpolate"
    id: str
    backed_type: str
    interpolate_factor: int
    precision: str = "auto"


class UpscaleModelClientInput(BaseModel):
    type: Literal["upscale"] = "upscale"
    id: str
    backed_type: str
    precision: str = "auto"


class EnhancementModelClientInput(BaseModel):
    type: Literal["enhancement"] = "enhancement"
    id: str
    backed_type: str
    precision: str = "auto"


ModelInputVariant = Annotated[
    Union[
        InterpolateModelClientInput,
        UpscaleModelClientInput,
        EnhancementModelClientInput,
    ],
    Field(discriminator="type"),
]
