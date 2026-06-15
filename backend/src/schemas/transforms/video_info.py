from typing import Optional

from src.schemas.domain.model import InterpolateModel, UpscaleModel
from src.schemas.domain.render import RenderSettings

# --- Assuming these are your imports from your domain/schemas ---
from src.schemas.domain.video_info import InputVideoInfo, OutputVideoInfo
from src.schemas.request.video_info import (
    InputVideoInfoClientInput,
    OutputVideoInfoClientInput,
)
from src.schemas.transforms.render import RenderSettingsTransformer
from src.utils.video_info import OpenCVInfo


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
    def to_domain(
        self,
        client_model: OutputVideoInfoClientInput,
        domain_input: InputVideoInfo,
        upscale_model: Optional[UpscaleModel],
        interpolate_model: Optional[InterpolateModel],
        hdr_mode: bool,
    ) -> OutputVideoInfo:

        return OutputVideoInfo(
            output_file=client_model.output_file,
            duration_seconds=domain_input.duration_seconds,  # pulled from input domain
            total_frames=domain_input.total_frames,  # pulled from input domain
            width=domain_input.width * upscale_model.scale
            if upscale_model
            else domain_input.width,  # (or scale based on self.upscale_model)
            height=domain_input.height * upscale_model.scale
            if upscale_model
            else domain_input.height,
            fps=domain_input.fps * interpolate_model.interpolate_factor
            if interpolate_model
            else domain_input.fps,  # (or modify based on self.interpolate_model)
            color_space=domain_input.color_space,
            pixel_format=domain_input.pixel_format,
            color_transfer=domain_input.color_transfer,
            color_primaries=domain_input.color_primaries,
            rotation=domain_input.rotation,
            bitrate=domain_input.bitrate,
            codec=domain_input.codec,
            is_hdr=domain_input.is_hdr and hdr_mode,
            bit_depth=domain_input.bit_depth,
        )
