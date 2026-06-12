from __future__ import annotations
from src.schemas.domain.model import InterpolateModel, UpscaleModel
from pydantic import BaseModel
from datetime import time


class RenderSettings(BaseModel):
    tiling_enabled: bool
    tilesize: int
    benchmark_mode: bool
    slow_mo_mode: bool
    hdr_mode: bool = False
    overwrite: bool = False
