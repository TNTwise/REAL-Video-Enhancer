from fastapi import APIRouter, Depends

from src.repos.model_repo import ModelRepo
from src.schemas.request import RenderSettingsClientInput
from src.schemas.transforms.model import (
    EnhancementModelTransformer,
    InterpolateModelTransformer,
    UpscaleModelTransformer,
)
from src.schemas.transforms.render import RenderSettingsTransformer
from src.schemas.transforms.video_info import (
    InputVideoInfoTransformer,
    OutputVideoInfoTransformer,
)
from src.services import RenderService

router = APIRouter(prefix="/render")


def get_model_repo() -> ModelRepo:
    repo = ModelRepo()
    repo.load()
    return repo


def get_input_video_transformer() -> InputVideoInfoTransformer:
    return InputVideoInfoTransformer()


def get_output_video_transformer() -> OutputVideoInfoTransformer:
    return OutputVideoInfoTransformer()


def get_interpolate_transformer(
    model_repo: ModelRepo = Depends(get_model_repo),
) -> InterpolateModelTransformer:
    return InterpolateModelTransformer(model_repo)


def get_upscale_transformer(
    model_repo: ModelRepo = Depends(get_model_repo),
) -> UpscaleModelTransformer:
    return UpscaleModelTransformer(model_repo)


def get_enhancement_transformer(
    model_repo: ModelRepo = Depends(get_model_repo),
) -> EnhancementModelTransformer:
    return EnhancementModelTransformer(model_repo)


def get_render_settings_transformer() -> RenderSettingsTransformer:
    return RenderSettingsTransformer()


def get_render_service(
    model_repo: ModelRepo = Depends(get_model_repo),
    input_video_tf: InputVideoInfoTransformer = Depends(get_input_video_transformer),
    output_video_tf: OutputVideoInfoTransformer = Depends(
        get_output_video_transformer
    ),
    interpolate_tf: InterpolateModelTransformer = Depends(get_interpolate_transformer),
    upscale_tf: UpscaleModelTransformer = Depends(get_upscale_transformer),
    enhancement_tf: EnhancementModelTransformer = Depends(get_enhancement_transformer),
    render_settings_tf: RenderSettingsTransformer = Depends(
        get_render_settings_transformer
    ),
) -> RenderService:
    return RenderService(
        model_repo=model_repo,
        input_video_transformer=input_video_tf,
        output_video_transformer=output_video_tf,
        interpolate_transformer=interpolate_tf,
        upscale_transformer=upscale_tf,
        enhancement_transformer=enhancement_tf,
        render_settings_transformer=render_settings_tf,
    )


@router.post("/start_render")
async def start_render(
    body: RenderSettingsClientInput,
    service: RenderService = Depends(get_render_service),
):
    return await service.start_render(body)
