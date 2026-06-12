from pydantic import BaseModel
from pathlib import Path

from src.schemas.request.video_info import VideoInfoClientInput
from src.utils.video_info import OpenCVInfo


class InputVideoInfo(BaseModel):
    input_file: str
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


class VideoInfoTransformer:
    def to_domain(self, client_model: VideoInfoClientInput) -> VideoInfo:
        OpenCVInfo()
