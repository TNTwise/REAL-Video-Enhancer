from __future__ import annotations

import asyncio
import os

from src.logic.handlers.backend.ncnn_handler import NCNNHandler
from src.logic.handlers.backend.pytorch_handler import TorchHandler
from src.logic.io.ffmpeg import (
    FFmpegRead,
    FFmpegWrite,
)
from src.logic.proxy.settings import PersistentSettingsProxy
from src.logic.render.backends.pytorch.TorchUtils import TorchUtils
from src.logic.render.backends.pytorch.UpscaleTorch import UpscalePytorch
from src.logic.render.methods.interpolate.interpolate_ncnn import InterpolateNCNN
from src.logic.render.methods.interpolate.interpolate_pytorch import InterpolatePyTorch
from src.logic.render.methods.upscale.upscale_pytorch import UpscalePyTorch
from src.logic.repos.model_repo import ModelRepo
from src.schemas.request import RenderSettingsClientInput
from src.schemas.transforms import (
    EnhancementModelTransformer,
    InputVideoInfoTransformer,
    InterpolateModelTransformer,
    OutputVideoInfoTransformer,
    RenderSettingsTransformer,
    UpscaleModelTransformer,
)
from src.utils.LogConfig import get_logger

logger = get_logger(__name__)

_background_tasks: set[asyncio.Task] = set()


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

        # ts all breaks di, but i dont care. This refactor is sucking the soul out of me.
        # TODO: Make this not as shit
        torch_utils = TorchUtils(
            input_video_info=input_settings,
            settings_proxy=persistent_settings,
            torch_handler=TorchHandler(),
        )

        interpolate_method = None
        upscale_method = None

        if interpolate_model:
            abs_path = self._model_repo.ensure(
                interpolate_model.id, interpolate_model.backend.type
            )
            interpolate_model.file_path = abs_path
            if interpolate_model.backend.type == "ncnn":
                ncnn_handler = NCNNHandler()
                interpolate_method = InterpolateNCNN(
                    ncnn_handler, interpolate_model, input_settings
                )
                interpolate_method._load()
            elif interpolate_model.backend.type == "pytorch":
                interpolate_method = InterpolatePyTorch(
                    torch_utils, interpolate_model, input_settings, persistent_settings
                )
                interpolate_method._load()

        if upscale_model:
            abs_path = self._model_repo.ensure(
                upscale_model.id, upscale_model.backend.type
            )
            upscale_model.file_path = abs_path
            if upscale_model.backend.type == "pytorch":
                upscale_method = UpscalePyTorch(
                    torch_utils, upscale_model, input_settings, persistent_settings
                )
                upscale_method._load()

        if input.enhancement_models:
            enhancement_models = []
            for enhancement_model in input.enhancement_models:
                if enhancement_model:
                    enhancement_models.append(
                        self.enhancement_tf.to_domain(enhancement_model)
                    )
        else:
            enhancement_models = None

        output_settings = self.output_video_tf.to_domain(
            client_model=input.output_video_info,
            domain_input=input_settings,
            interpolate_model=interpolate_model,
            upscale_model=upscale_model,
        )

        render_settings = self._render_settings_tf.to_domain(
            input,
            input_video_info=input_settings,
            output_video_info=output_settings,
            upscale_model=upscale_model,
            interpolate_model=interpolate_model,
            enhancement_models=enhancement_models,
        )

        read_buffer = FFmpegRead(
            render_settings,
            input_settings,
            output_settings,
            persistent_settings,
            torch_utils,
        )

        write_buffer = FFmpegWrite(
            render_settings, input_settings, output_settings, persistent_settings
        )

        task = asyncio.create_task(
            self._background_render(
                render_settings=render_settings,
                persistent_settings=persistent_settings,
                read_buffer=read_buffer,
                write_buffer=write_buffer,
                interpolate_method=interpolate_method,
                upscale_method=upscale_method,
            )
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

        return {
            "status": "started",
            "message": "Render started",
        }

    async def _background_render(
        self,
        render_settings,
        persistent_settings,
        read_buffer,
        write_buffer,
        interpolate_method,
        upscale_method,
    ):
        try:
            await self._render_proxy.render(
                render_settings=render_settings,
                persistent_settings=persistent_settings,
                read_buffer=read_buffer,
                write_buffer=write_buffer,
                interpolate_method=interpolate_method,
                upscale_method=upscale_method,
            )
        except Exception:
            logger.exception("Background render failed")
