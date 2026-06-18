from fastapi import APIRouter, Depends

from src.logic.io.ffmpeg import FFmpegWrite
from src.logic.proxy.settings import PersistentSettingsProxy
from src.logic.render.render_video import RenderProxy
from src.logic.repos.model_repo import ModelRepo
from src.logic.services import RenderService
from src.schemas.request import RenderSettingsClientInput
from src.schemas.transforms.model import (
    EnhancementModelTransformer,
    InterpolateModelTransformer,
    UpscaleModelTransformer,
)
from src.schemas.transforms.precision import PrecisionTransform
from src.schemas.transforms.render import RenderSettingsTransformer
from src.schemas.transforms.video_info import (
    InputVideoInfoTransformer,
    OutputVideoInfoTransformer,
)

router = APIRouter(prefix="/render")

_current_write_buffer: FFmpegWrite | None = None


def get_current_write_buffer() -> FFmpegWrite | None:
    return _current_write_buffer


def set_current_write_buffer(buffer: FFmpegWrite | None):
    global _current_write_buffer
    _current_write_buffer = buffer


def get_render_proxy() -> RenderProxy:
    return RenderProxy()


def get_model_repo() -> ModelRepo:
    repo = ModelRepo()
    repo.load()
    return repo


def get_input_video_transformer() -> InputVideoInfoTransformer:
    return InputVideoInfoTransformer()


def get_output_video_transformer() -> OutputVideoInfoTransformer:
    return OutputVideoInfoTransformer()


def get_precision_transformer() -> PrecisionTransform:
    return PrecisionTransform()


def get_interpolate_transformer(
    model_repo: ModelRepo = Depends(get_model_repo),
    precision_tf: PrecisionTransform = Depends(get_precision_transformer),
) -> InterpolateModelTransformer:
    return InterpolateModelTransformer(model_repo, precision_tf)


def get_upscale_transformer(
    model_repo: ModelRepo = Depends(get_model_repo),
    precision_tf: PrecisionTransform = Depends(get_precision_transformer),
) -> UpscaleModelTransformer:
    return UpscaleModelTransformer(model_repo, precision_tf)


def get_enhancement_transformer(
    model_repo: ModelRepo = Depends(get_model_repo),
    precision_tf: PrecisionTransform = Depends(get_precision_transformer),
) -> EnhancementModelTransformer:
    return EnhancementModelTransformer(model_repo, precision_tf)


def get_render_settings_transformer() -> RenderSettingsTransformer:
    return RenderSettingsTransformer()


def get_persistent_settings() -> PersistentSettingsProxy:
    return PersistentSettingsProxy()


def get_render_service(
    model_repo: ModelRepo = Depends(get_model_repo),
    input_video_tf: InputVideoInfoTransformer = Depends(get_input_video_transformer),
    output_video_tf: OutputVideoInfoTransformer = Depends(get_output_video_transformer),
    interpolate_tf: InterpolateModelTransformer = Depends(get_interpolate_transformer),
    upscale_tf: UpscaleModelTransformer = Depends(get_upscale_transformer),
    enhancement_tf: EnhancementModelTransformer = Depends(get_enhancement_transformer),
    render_settings_tf: RenderSettingsTransformer = Depends(
        get_render_settings_transformer
    ),
    render_proxy: RenderProxy = Depends(get_render_proxy),
) -> RenderService:
    return RenderService(
        model_repo=model_repo,
        input_video_transformer=input_video_tf,
        output_video_transformer=output_video_tf,
        interpolate_transformer=interpolate_tf,
        upscale_transformer=upscale_tf,
        enhancement_transformer=enhancement_tf,
        render_settings_transformer=render_settings_tf,
        render_proxy=render_proxy,
    )


@router.post("/start_render")
async def start_render(
    body: RenderSettingsClientInput,
    service: RenderService = Depends(get_render_service),
    settings: PersistentSettingsProxy = Depends(get_persistent_settings),
):
    return await service.start_render(body, settings)
