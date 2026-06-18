from fastapi import APIRouter, Depends

from src.logic.proxy import PersistentSettingsProxy
from src.schemas.request import Setting

router = APIRouter(prefix="/PersistentSettingsProxy")


# FastAPI example
def get_user_service() -> PersistentSettingsProxy:
    return PersistentSettingsProxy()


@router.post("/update_setting")
def update_setting(
    body: Setting, service: PersistentSettingsProxy = Depends(get_user_service)
):
    return service.write_setting(body.setting, str(body.value))


@router.get("/get_setting_value")
def get_setting_value(
    setting: str, service: PersistentSettingsProxy = Depends(get_user_service)
):
    return service.get_setting(setting)


@router.get("/get_allowed_options")
def get_allowed_options(
    setting: str, service: PersistentSettingsProxy = Depends(get_user_service)
):
    return service.get_allowed_options(setting)
