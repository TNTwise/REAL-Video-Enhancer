from src.proxy.backends.pytorch.InterpolateArchs.GIMM.GIMM import model
from src.schemas.domain.render import RenderSettings
from src.schemas.request.render import RenderSettingsClientInput
from src.schemas.transforms.model import (
    EnhancementModelTransformer,
    InterpolateModelTransformer,
    UpscaleModelTransformer,
)
from src.schemas.transforms.video_info import (
    InputVideoInfoTransformer,
    OutputVideoInfoTransformer,
)


class RenderSettingsTransformer:
    def __init__(
        self,
        input_video_transformer: InputVideoInfoTransformer,
        output_video_transformer: OutputVideoInfoTransformer,
        interpolate_transformer: InterpolateModelTransformer,
        upscale_transformer: UpscaleModelTransformer,
        enhancement_transformer: EnhancementModelTransformer,
    ):
        # Injecting the sub-transformers via the constructor
        self.interpolate_tf = interpolate_transformer
        self.upscale_tf = upscale_transformer
        self.enhancement_tf = enhancement_transformer
        self.input_video_tf = input_video_transformer
        self.output_video_tf = output_video_transformer

    def to_domain(self, client_input: RenderSettingsClientInput) -> RenderSettings:
        # 1. Transform the simple fields directly
        domain_tiling = client_input.tiling_enabled
        domain_slow_mo_mode = client_input.slow_mo_mode
        domain_tiling_size = client_input.tilesize
        domain_bechmark_mode = client_input.benchmark_mode

        # 2. Delegate the complex fields inline (Pythonic conditional assignment)
        domain_interpolate = (
            self.interpolate_tf.to_domain(client_input.interpolate_model)
            if client_input.interpolate_model
            else None
        )

        domain_upscale = (
            self.upscale_tf.to_domain(client_input.upscale_model)
            if client_input.upscale_model
            else None
        )

        # 3. Handle collections seamlessly using a list comprehension or empty list fallback
        domain_enhancements = [
            self.enhancement_tf.to_domain(model)
            for model in (client_input.enhancement_models or [])
        ]

        # Inter-dependent field transformations
        domain_input_video_info = self.input_video_tf.to_domain(
            client_input.input_video_info
        )
        domain_output_video_info = self.output_video_tf.to_domain(
            client_input.output_video_info, domain_input_video_info
        )

        # 4. Construct and return your domain object
        return RenderSettings(
            input_video_info=domain_input_video_info,
            output_video_info=domain_output_video_info,
            tiling_enabled=domain_tiling,
            slow_mo_mode=domain_slow_mo_mode,
            tilesize=domain_tiling_size,
            interpolate_model=domain_interpolate,
            upscale_model=domain_upscale,
            enhancement_models=domain_enhancements,
            benchmark_mode=domain_bechmark_mode,
        )
