from src.schemas.domain.model import EnhancementModel, InterpolateModel, UpscaleModel
from src.schemas.domain.render import RenderSettings
from src.schemas.domain.video_info import InputVideoInfo, OutputVideoInfo
from src.schemas.request.render import RenderSettingsClientInput


class RenderSettingsTransformer:
    def to_domain(
        self,
        client_input: RenderSettingsClientInput,
        input_video_info: InputVideoInfo,
        output_video_info: OutputVideoInfo,
        interpolate_model: InterpolateModel | None,
        upscale_model: UpscaleModel | None,
        enhancement_models: list[EnhancementModel | None] | None,
    ) -> RenderSettings:
        # 1. Transform the simple fields directly
        domain_tiling = client_input.tiling_enabled
        domain_slow_mo_mode = client_input.slow_mo_mode
        domain_tiling_size = client_input.tilesize
        domain_bechmark_mode = client_input.benchmark_mode

        # 4. Construct and return your domain object
        return RenderSettings(
            input_video_info=input_video_info,
            output_video_info=output_video_info,
            tiling_enabled=domain_tiling,
            slow_mo_mode=domain_slow_mo_mode,
            tilesize=domain_tiling_size,
            interpolate_model=interpolate_model,
            upscale_model=upscale_model,
            enhancement_models=enhancement_models,
            benchmark_mode=domain_bechmark_mode,
        )
