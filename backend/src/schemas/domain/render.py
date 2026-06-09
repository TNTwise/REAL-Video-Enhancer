from __future__ import annotations
from backend.src.schemas.domain.model import InterpolateModel, UpscaleModel
from pydantic import BaseModel


class RenderSettings(BaseModel):
    video_path: str
    default_output_path_override: str | None = None
    interpolate_model: InterpolateModel | None = None
    upscale_model: UpscaleModel | None = None
    enhancement_models: list[UpscaleModel] | None = None
