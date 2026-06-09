from pydantic import BaseModel
from typing import Any

class Setting(BaseModel):
    setting: str
    value: Any