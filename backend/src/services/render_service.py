from __future__ import annotations

import os

from src.proxy.io_buffers.ffmpeg_proxy import (
    FFmpegRead,
    FFmpegWrite,
)
from src.proxy.settings import PersistentSettingsProxy
from src.repos.model_repo import ModelRepo
from src.schemas.request import RenderSettingsClientInput
from src.schemas.transforms import (
    EnhancementModelTransformer,
    InputVideoInfoTransformer,
    InterpolateModelTransformer,
    OutputVideoInfoTransformer,
    RenderSettingsTransformer,
    UpscaleModelTransformer,
)


# TODO: the render service should return immidiately, not hang the server up on a single thread. Mess with asyncio shit to fix this
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
        render_proxy,
    ):
        # Injecting the sub-transformers via the constructor
        self.interpolate_tf = interpolate_transformer
        self.upscale_tf = upscale_transformer
        self.enhancement_tf = enhancement_transformer
        self.input_video_tf = input_video_transformer
        self.output_video_tf = output_video_transformer
        self._model_repo = model_repo
        self._render_settings_tf = render_settings_transformer
        self._render_proxy = render_proxy

    def _resolve_model(self, model_id: str, backend_type: str):
        for model in self._model_repo.get_by_backend(backend_type):
            if model.id == model_id and model.backend.type == backend_type:
                return model
        raise ValueError(f"Model '{model_id}' not found for backend '{backend_type}'")

    async def start_render(
        self,
        input: RenderSettingsClientInput,
        persistent_settings: PersistentSettingsProxy,
    ):
        if not os.path.isfile(input.input_video_info.input_file):
            raise FileNotFoundError(
                f"Input file does not exist: {input.input_video_info.input_file}"
            )

        input_settings = self.input_video_tf.to_domain(input.input_video_info)

        interpolate_model = (
            self.interpolate_tf.to_domain(input.interpolate_model)
            if input.interpolate_model
            else None
        )

        upscale_model = (
            self.upscale_tf.to_domain(input.upscale_model)
            if input.upscale_model
            else None
        )
        if input.enhancement_models:
            enhancement_models = []
            for enhancement_model in input.enhancement_models:
                if enhancement_model:
                    enhancement_models.append(
                        self.enhancement_tf.to_domain(enhancement_model)
                    )
        else:
            enhancement_models = None

        hdr_mode = persistent_settings.auto_hdr_mode == "True"

        output_settings = self.output_video_tf.to_domain(
            client_model=input.output_video_info,
            domain_input=input_settings,
            interpolate_model=interpolate_model,
            upscale_model=upscale_model,
            hdr_mode=hdr_mode,
        )

        render_settings = self._render_settings_tf.to_domain(
            input,
            input_video_info=input_settings,
            output_video_info=output_settings,
            upscale_model=upscale_model,
            interpolate_model=interpolate_model,
            enhancement_models=enhancement_models,
        )

        # TODO make this not so shit, breaks DI

        read_buffer = FFmpegRead(
            render_settings, input_settings, output_settings, persistent_settings
        )

        write_buffer = FFmpegWrite(
            render_settings, input_settings, output_settings, persistent_settings
        )

        await self._render_proxy.render(
            render_settings=render_settings,
            persistent_settings=persistent_settings,
            read_buffer=read_buffer,
            write_buffer=write_buffer,
        )
