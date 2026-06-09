from __future__ import annotations
from backend.src.schemas.domain.model import InterpolateModel, UpscaleModel
from pydantic import BaseModel


class RenderSettings(BaseModel):
    video_path: str
    default_output_path_override: str | None
    interpolate_model: InterpolateModel
    upscale_model: UpscaleModel
    enhancement_models: list[UpscaleModel]
