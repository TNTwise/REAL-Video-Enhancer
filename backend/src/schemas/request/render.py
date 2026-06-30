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


# This is for any setting set by the user without being saved, and not saved in settings.
class RenderSettingsClientInput(BaseModel):
    input_video_info: InputVideoInfoClientInput
    output_video_info: OutputVideoInfoClientInput
    benchmark_mode: bool
    slow_mo_mode: bool
    interpolate_model: InterpolateModelClientInput | None = None
    upscale_model: UpscaleModelClientInput | None = None
    enhancement_models: list[EnhancementModelClientInput | None] | None = None
    start_time: float | None = None
    end_time: float | None = None
