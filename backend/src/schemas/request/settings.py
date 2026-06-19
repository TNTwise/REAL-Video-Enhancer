from typing import Any

from pydantic import BaseModel


class Setting(BaseModel):
    setting: str
    value: Any
