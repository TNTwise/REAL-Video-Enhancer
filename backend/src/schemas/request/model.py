from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal, Annotated, Union

class InterpolateModelClientInput(BaseModel):
    type: Literal["interpolate"] = "interpolate"
    variant: str


class UpscaleModelClientInput(BaseModel):
    type: Literal["upscale"] = "upscale"
    variant: str

class EnhancementModelClientInput(BaseModel):
    type: Literal["enhancement"] = "enhancement"
    variant: str
    
ModelInputVariant = Annotated[Union[InterpolateModelClientInput, UpscaleModelClientInput, EnhancementModelClientInput], Field(discriminator="type")]
