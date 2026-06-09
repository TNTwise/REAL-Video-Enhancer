from __future__ import annotations
from backend.src.schemas.request.model import InterpolateModelClientInput, UpscaleModelClientInput
from pydantic import BaseModel

class RenderSettingsClientInput(BaseModel):
    video_path: str
    default_output_path_override: str | None
    interpolate_model: InterpolateModelClientInput
    upscale_model: UpscaleModelClientInput
    enhancement_models: list[UpscaleModelClientInput]