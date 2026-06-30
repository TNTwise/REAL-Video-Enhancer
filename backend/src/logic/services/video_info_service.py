import os

from src.schemas.domain.video_info import InputVideoInfo, OutputVideoInfo
from src.schemas.request import (
    InputVideoInfoClientInput,
    OutputVideoInfoClientInput,
)
from src.schemas.request.model import (
    InterpolateModelClientInput,
    UpscaleModelClientInput,
)
from src.schemas.transforms import (
    InputVideoInfoTransformer,
    InterpolateModelTransformer,
    OutputVideoInfoTransformer,
    RenderSettingsTransformer,
    UpscaleModelTransformer,
)


class VideoInfoService:
    def __init__(
        self,
        input_video_info_transformer: InputVideoInfoTransformer,
        output_video_info_transformer: OutputVideoInfoTransformer,
        render_settings_transformer: RenderSettingsTransformer,
        interpolate_model_transformer: InterpolateModelTransformer,
        upscale_model_transformer: UpscaleModelTransformer,
    ):
        self._input_video_info_transformer = input_video_info_transformer
        self._output_video_info_transformer = output_video_info_transformer
        self._interpolate_model_transformer = interpolate_model_transformer
        self._upscale_model_transformer = upscale_model_transformer

    def get_input_video_info(self, body: InputVideoInfoClientInput) -> InputVideoInfo:
        if not os.path.isfile(body.input_file):
            raise FileNotFoundError(f"Input file not found: {body.input_file}")

        return self._input_video_info_transformer.to_domain(body)

    def get_output_video_info(
        self,
        body: OutputVideoInfoClientInput,
        input_video_info_client_input: InputVideoInfoClientInput,
        interpolate_model_client_input: InterpolateModelClientInput | None,
        upscale_model_client_input: UpscaleModelClientInput | None,
    ) -> OutputVideoInfo:
        if os.path.isfile(body.output_file):
            raise FileExistsError("File already exists!")

        input_settings = self._input_video_info_transformer.to_domain(
            input_video_info_client_input
        )
        interpolate_model = (
            self._interpolate_model_transformer.to_domain(
                interpolate_model_client_input
            )
            if interpolate_model_client_input
            else None
        )

        upscale_model = (
            self._upscale_model_transformer.to_domain(upscale_model_client_input)
            if upscale_model_client_input
            else None
        )

        return self._output_video_info_transformer.to_domain(
            client_model=body,
            domain_input=input_settings,
            interpolate_model=interpolate_model,
            upscale_model=upscale_model,
        )
