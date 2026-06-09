from backend.src.schemas import RenderSettings
from backend.src.schemas import RenderSettingsClientInput


class RenderService:
    def _renderclientinput_to_rendersettings(
        self, input: RenderSettingsClientInput
    ) -> RenderSettings:
        return RenderSettings(width=input.width, height=input.height, fps=input.fps)
