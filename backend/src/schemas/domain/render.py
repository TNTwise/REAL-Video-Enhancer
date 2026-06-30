from __future__ import annotations

from pydantic import BaseModel

from src.schemas.domain.model import EnhancementModel, InterpolateModel, UpscaleModel
from src.schemas.domain.video_info import InputVideoInfo, OutputVideoInfo


# This is for any setting set by the user, and not saved in settings.
class RenderSettings(BaseModel):
    input_video_info: InputVideoInfo
    output_video_info: OutputVideoInfo
    benchmark_mode: bool
    slow_mo_mode: bool
    overwrite: bool = False
    interpolate_model: InterpolateModel | None
    upscale_model: UpscaleModel | None
    enhancement_models: list[EnhancementModel | None] | None
