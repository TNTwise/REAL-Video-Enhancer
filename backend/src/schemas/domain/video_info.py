from pydantic import BaseModel
from typing import Optional

# --- Assuming these are your imports from your domain/schemas ---
from src.schemas.request.video_info import (
    InputVideoInfoClientInput,
    OutputVideoInfoClientInput,
)
from src.utils.video_info import OpenCVInfo
from src.schemas.domain.render import RenderSettings
from src.schemas.domain.model import InterpolateModel, UpscaleModel


# 1. Domain Models (from your code)
class InputVideoInfo(BaseModel):
    input_file: str
    duration_seconds: float
    total_frames: int
    width: int
    height: int
    fps: float
    color_space: str | None = None
    pixel_format: str
    color_transfer: str | None = None
    color_primaries: str | None = None
    rotation: float
    bitrate: int
    codec: str
    is_hdr: bool
    bit_depth: int


class OutputVideoInfo(BaseModel):
    output_file: str
    duration_seconds: float
    total_frames: int
    width: int
    height: int
    fps: float
    color_space: str
    pixel_format: str
    color_transfer: str
    color_primaries: str
    rotation: float
    bitrate: int
    codec: str
    is_hdr: bool
    bit_depth: int


# 2. Input Transformer (Stateless - stays simple)
class InputVideoInfoTransformer:
    def to_domain(self, client_model: InputVideoInfoClientInput) -> InputVideoInfo:
        info = OpenCVInfo(client_model)
        return InputVideoInfo(
            input_file=client_model.input_file,
            duration_seconds=info.get_duration_seconds(),
            total_frames=info.get_total_frames(),
            width=info.input_width,
            height=info.input_height,
            fps=info.input_fps,
            color_space=info.color_space,
            pixel_format=info.pixel_format,
            color_transfer=info.color_transfer,
            color_primaries=info.color_primaries,
            rotation=info.rotation,
            bitrate=info.bitrate,
            codec=info.codec,
            is_hdr=info.is_hdr,
            bit_depth=info.bit_depth,
        )


class OutputVideoInfoTransformer:
    def __init__(
        self,
        render_settings: RenderSettings,
        interpolate_model: Optional[InterpolateModel] = None,
        upscale_model: Optional[UpscaleModel] = None,
    ):
        self.render_settings = render_settings
        self.interpolate_model = interpolate_model
        self.upscale_model = upscale_model

    def to_domain(
        self, client_model: OutputVideoInfoClientInput, domain_input: InputVideoInfo
    ) -> OutputVideoInfo:

        return OutputVideoInfo(
            output_file=client_model.output_file,
            duration_seconds=domain_input.duration_seconds,  # pulled from input domain
            total_frames=domain_input.total_frames,  # pulled from input domain
            width=domain_input.width * self.upscale_model.scale
            if self.upscale_model
            else 1,  # (or scale based on self.upscale_model)
            height=domain_input.height,
            fps=domain_input.fps * self.interpolate_model.interpolate_factor
            if self.interpolate_model
            else 1,  # (or modify based on self.interpolate_model)
            color_space=domain_input.color_space,
            pixel_format=domain_input.pixel_format,
            color_transfer=domain_input.color_transfer,
            color_primaries=domain_input.color_primaries,
            rotation=domain_input.rotation,
            bitrate=domain_input.bitrate,
            codec=domain_input.codec,
            is_hdr=domain_input.is_hdr and self.render_settings.hdr_mode,
            bit_depth=domain_input.bit_depth,
        )
