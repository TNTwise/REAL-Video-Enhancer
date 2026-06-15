from __future__ import annotations

import os

from src.proxy.render_video import Render
from src.repos.model_repo import ModelRepo
from src.schemas.transforms import (
    EnhancementModelTransformer,
    InputVideoInfoTransformer,
    InterpolateModelTransformer,
    OutputVideoInfoTransformer,
    RenderSettingsTransformer,
    UpscaleModelTransformer,
)


class RenderService:
    def __init__(
        self,
        model_repo: ModelRepo,
        input_video_transformer: InputVideoInfoTransformer,
        output_video_transformer: OutputVideoInfoTransformer,
        interpolate_transformer: InterpolateModelTransformer,
        upscale_transformer: UpscaleModelTransformer,
        enhancement_transformer: EnhancementModelTransformer,
        render_settings_transformer: RenderSettingsTransformer,
    ):
        # Injecting the sub-transformers via the constructor
        self.interpolate_tf = interpolate_transformer
        self.upscale_tf = upscale_transformer
        self.enhancement_tf = enhancement_transformer
        self.input_video_tf = input_video_transformer
        self.output_video_tf = output_video_transformer
        self._model_repo = model_repo

    def _resolve_model(self, model_id: str, backend_type: str):
        for model in self._model_repo.get_by_backend(backend_type):
            if model.id == model_id and model.backend.type == backend_type:
                return model
        raise ValueError(f"Model '{model_id}' not found for backend '{backend_type}'")

    async def start_render(self, input: RenderSettingsClientInput):
        if not os.path.isfile(input.input_video_info.input_file):
            raise FileNotFoundError(
                f"Input file does not exist: {input.input_video_info.input_file}"
            )
        settings = self._renderclientinput_to_rendersettings(input)
        await self._render_proxy.render(read_buffer, self.write_buffer)
