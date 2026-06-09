from fastapi import APIRouter, Depends
from src.schemas import Setting
from src.services import Settings

router = APIRouter(prefix="/settings")
# FastAPI example
def get_user_service() -> Settings:
    return Settings()

@router.post("/update_setting")
def update_setting(body: Setting, service: Settings = Depends(get_user_service)):
    return service.write_setting(body.setting, str(body.value))


@router.get("/get_setting_value")
def get_setting_value(setting: str, service: Settings = Depends(get_user_service)):
    return service.get_setting_value(setting)


@router.get("/get_allowed_options")
def get_allowed_options(setting: str, service: Settings = Depends(get_user_service)):
    return service.get_allowed_options(setting)
