from fastapi import APIRouter
from src.models.settings import Setting
from src.settings import settings

router = APIRouter(prefix="/settings")

@router.put("/write_setting")
def update_setting(body: Setting):
    settings.write_setting(body.setting, str(body.value))

@router.get("/get_setting")
def get_setting_value(setting: str):
    settings.get_setting_value(setting)

@router.get("/get_allowed_options")
def get_allowed_options(setting: str):
    settings.get_allowed_options(setting)