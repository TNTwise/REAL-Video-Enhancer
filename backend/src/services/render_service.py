from __future__ import annotations

from proxy.io_buffers.ffmpeg_proxy import ReadBuffer, WriteBuffer
from src.schemas import RenderSettings, RenderSettingsClientInput
from src.repos.model_repo import ModelRepo
from src.proxy.video_info_proxy import OpenCVInfo
from src.proxy.render_video import Render


class RenderService:
    def __init__(
        self,
        model_repo: ModelRepo,
        video_info_proxy: OpenCVInfo,
        render_proxy: Render,
        read_buffer: ReadBuffer,
        write_buffer: WriteBuffer,
    ):
        self._model_repo = model_repo
        self._video_info_service = video_info_proxy
        self._render_proxy = render_proxy

        self._write_buffer = write_buffer
        self._read_buffer = read_buffer

    def _renderclientinput_to_rendersettings(
        self, input: RenderSettingsClientInput
    ) -> RenderSettings:
        """Resolve client-side model IDs into full domain models from the repo."""
        interpolate_model = self._resolve_model(
            input.interpolate_model.id, input.interpolate_model.backed_type
        )
        upscale_model = self._resolve_model(
            input.upscale_model.id, input.upscale_model.backed_type
        )
        enhancement_models = [
            self._resolve_model(m.id, m.backed_type) for m in input.enhancement_models
        ]

        return RenderSettings(
            video_path=input.video_path,
            default_output_path_override=input.default_output_path_override,
            interpolate_model=interpolate_model,
            upscale_model=upscale_model,
            enhancement_models=enhancement_models,
        )

    def _resolve_model(self, model_id: str, backend_type: str):
        for model in self._model_repo.get_by_backend(backend_type):
            if model.id == model_id and model.backend.type == backend_type:
                return model
        raise ValueError(f"Model '{model_id}' not found for backend '{backend_type}'")

    async def start_render(self, input: RenderSettingsClientInput):
        settings = self._renderclientinput_to_rendersettings(input)
        await self._render_proxy.render(self.read_buffer, self.write_buffer)
