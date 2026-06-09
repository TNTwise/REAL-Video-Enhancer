from __future__ import annotations
from src.schemas.domain.model import InterpolateModel, UpscaleModel
from pydantic import BaseModel
from datetime import time

class RenderSettings(BaseModel):
    video_path: str
    tiling_enabled: bool
    tilesize: int
    interpolate_times: int
    benchmark_mode: bool
    slow_mo_mode: bool
    hdr_mode: bool = False
    start_time: time | None = None
    end_time: time | None = None
    default_output_path_override: str | None = None
    interpolate_model: InterpolateModel | None = None
    upscale_model: UpscaleModel | None = None
    enhancement_models: list[UpscaleModel] | None = None
