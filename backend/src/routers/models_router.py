from fastapi import APIRouter, Depends, Query

from src.logic.services.model_service import ModelService

router = APIRouter(prefix="/models", tags=["Models"])


def get_model_service() -> ModelService:
    return ModelService()


@router.get("/interpolate")
def get_interpolate_models(
    backend_id: str | None = Query(
        None, description="Filter by backend: ncnn, pytorch, tensorrt"
    ),
    service: ModelService = Depends(get_model_service),
):
    return service.get_interpolate_models(backend_id)


@router.get("/upscale")
def get_upscale_models(
    backend_id: str | None = Query(
        None, description="Filter by backend: ncnn, pytorch, tensorrt"
    ),
    service: ModelService = Depends(get_model_service),
):
    return service.get_upscale_models(backend_id)


@router.get("/enhancement")
def get_enhancement_models(
    backend_id: str | None = Query(
        None, description="Filter by backend: ncnn, pytorch, tensorrt"
    ),
    service: ModelService = Depends(get_model_service),
):
    return service.get_enhancement_models(backend_id)
