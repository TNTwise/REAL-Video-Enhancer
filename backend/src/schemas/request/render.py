from __future__ import annotations

from pydantic import BaseModel

from src.schemas.request.model import (
    EnhancementModelClientInput,
    InterpolateModelClientInput,
    UpscaleModelClientInput,
)
from src.schemas.request.video_info import (
    InputVideoInfoClientInput,
    OutputVideoInfoClientInput,
)


class RenderSettingsClientInput(BaseModel):
    input_video_info: InputVideoInfoClientInput
    output_video_info: OutputVideoInfoClientInput
    tiling_enabled: bool
    tilesize: int
    benchmark_mode: bool
    slow_mo_mode: bool
    interpolate_model: InterpolateModelClientInput | None
    upscale_model: UpscaleModelClientInput | None
    enhancement_models: list[EnhancementModelClientInput | None] | None
    start_time: float | None = None
    end_time: float | None = None
