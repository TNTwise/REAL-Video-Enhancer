from fastapi import APIRouter, Depends
from src.schemas import RenderSettingsClientInput
from src.services import RenderService
from src.repos import ModelRepo
from src.
router = APIRouter(prefix="/render")


# FastAPI example
def get_user_service() -> RenderService:
    return RenderService()


@router.post("/start_render")
async def start_render(
    body: RenderSettingsClientInput, service: RenderService = Depends(get_user_service)
):
    return await service.start_render(body)
