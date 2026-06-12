from __future__ import annotations
from src.schemas.request.model import (
    InterpolateModelClientInput,
    UpscaleModelClientInput,
    EnhancementModelClientInput,
)
from pydantic import BaseModel


class RenderSettingsClientInput(BaseModel):
    tiling_enabled: bool
    tilesize: int
    benchmark_mode: bool
    slow_mo_mode: bool
    interpolate_model: InterpolateModelClientInput
    upscale_model: UpscaleModelClientInput
    enhancement_models: list[EnhancementModelClientInput]
    hdr_mode: bool = False
    start_time: float | None = None
    end_time: float | None = None
