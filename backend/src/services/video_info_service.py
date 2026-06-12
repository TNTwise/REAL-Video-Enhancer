import os

from src.schemas.domain.video_info import InputVideoInfo, InputVideoInfoTransformer
from src.schemas.request.video_info import InputVideoInfoClientInput
from src.utils.video_info import OpenCVInfo


class VideoInfoService:
    def get_input_video_info(self, body: InputVideoInfoClientInput) -> InputVideoInfo:
        if not os.path.isfile(body.input_file):
            raise FileNotFoundError(f"Input file not found: {body.input_file}")

        transformer = InputVideoInfoTransformer()
        return transformer.to_domain(body)
