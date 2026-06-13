from pydantic import BaseModel


# 1. Domain Models (from your code)
class InputVideoInfo(BaseModel):
    input_file: str
    duration_seconds: float
    total_frames: int
    width: int
    height: int
    fps: float
    color_space: str | None = None
    pixel_format: str | None = None
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
    color_space: str | None = None
    pixel_format: str | None = None
    color_transfer: str | None = None
    color_primaries: str | None = None
    rotation: float
    bitrate: int
    codec: str
    is_hdr: bool
    bit_depth: int
