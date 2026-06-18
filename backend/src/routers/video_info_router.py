from fastapi import APIRouter, Depends, HTTPException, status

from src.logic.repos import ModelRepo
from src.logic.services.video_info_service import VideoInfoService
from src.schemas.domain.video_info import OutputVideoInfo
from src.schemas.request import OutputVideoInfoClientInput
from src.schemas.request.model import (
    InterpolateModelClientInput,
    UpscaleModelClientInput,
)
from src.schemas.request.video_info import InputVideoInfoClientInput
from src.schemas.transforms import (
    InputVideoInfoTransformer,
    InterpolateModelTransformer,
    OutputVideoInfoTransformer,
    RenderSettingsTransformer,
    UpscaleModelTransformer,
)

router = APIRouter(prefix="/video")

# Singleton model repo instance
_model_repo = ModelRepo()
_model_repo.load()


def get_input_video_info_transformer() -> InputVideoInfoTransformer:
    return InputVideoInfoTransformer()


def get_output_video_info_transformer() -> OutputVideoInfoTransformer:
    return OutputVideoInfoTransformer()


def get_render_settings_transformer() -> RenderSettingsTransformer:
    return RenderSettingsTransformer()


def get_interpolate_model_transformer() -> InterpolateModelTransformer:
    return InterpolateModelTransformer(model_repo=_model_repo)


def get_upscale_model_transformer() -> UpscaleModelTransformer:
    return UpscaleModelTransformer(model_repo=_model_repo)


def get_video_info_service(
    input_video_info_transformer: InputVideoInfoTransformer = Depends(
        get_input_video_info_transformer
    ),
    output_video_info_transformer: OutputVideoInfoTransformer = Depends(
        get_output_video_info_transformer
    ),
    render_settings_transformer: RenderSettingsTransformer = Depends(
        get_render_settings_transformer
    ),
    interpolate_model_transformer: InterpolateModelTransformer = Depends(
        get_interpolate_model_transformer
    ),
    upscale_model_transformer: UpscaleModelTransformer = Depends(
        get_upscale_model_transformer
    ),
) -> VideoInfoService:

    return VideoInfoService(
        input_video_info_transformer=input_video_info_transformer,
        output_video_info_transformer=output_video_info_transformer,
        render_settings_transformer=render_settings_transformer,
        interpolate_model_transformer=interpolate_model_transformer,
        upscale_model_transformer=upscale_model_transformer,
    )


@router.get("/get_input_info")
def get_input_video_info(
    body: InputVideoInfoClientInput,
    service: VideoInfoService = Depends(get_video_info_service),
):
    return service.get_input_video_info(body)


@router.post("/video-info")
async def output_video_info(
    body: OutputVideoInfoClientInput,
    input_video: InputVideoInfoClientInput,
    interpolate: InterpolateModelClientInput | None = None,
    upscale: UpscaleModelClientInput | None = None,
    hdr_mode: bool = False,
    video_service: VideoInfoService = Depends(get_video_info_service),
) -> OutputVideoInfo:
    try:
        # Call your function
        return video_service.get_output_video_info(
            body=body,
            input_video_info_client_input=input_video,
            interpolate_model_client_input=interpolate,
            upscale_model_client_input=upscale,
            hdr_mode=hdr_mode,
        )
    except FileExistsError as e:
        # Map the Python error to a 409 Conflict or 400 Bad Request
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception:
        # Catch-all for unexpected transformer or system errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the video info.",
        )
