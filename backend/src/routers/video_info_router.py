from fastapi import APIRouter, Depends
from src.schemas.request.video_info import InputVideoInfoClientInput
from src.services.video_info_service import VideoInfoService

router = APIRouter(prefix="/video")


def get_video_info_service() -> VideoInfoService:
    return VideoInfoService()


@router.post("/get_input_info")
def get_input_video_info(
    body: InputVideoInfoClientInput,
    service: VideoInfoService = Depends(get_video_info_service),
):
    return service.get_input_video_info(body)
