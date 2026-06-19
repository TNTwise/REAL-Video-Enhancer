from src.logic.proxy import PersistentSettingsProxy
from src.logic.repos import ModelRepo
from src.schemas.domain import EnhancementModel, InterpolateModel, UpscaleModel
from src.schemas.domain.backend import PyTorchBackend
from src.schemas.repo.model import (
    EnhancementModelRepo,
    InterpolateModelRepo,
    UpscaleModelRepo,
)
from src.schemas.request.model import (
    EnhancementModelClientInput,
    InterpolateModelClientInput,
    UpscaleModelClientInput,
)
from src.schemas.transforms.precision import PrecisionTransform


class InterpolateModelTransformer:
    def __init__(
        self,
        model_repo: ModelRepo,
        precision_transform: PrecisionTransform,
        settings: PersistentSettingsProxy,
    ):
        self._model_repo = model_repo
        self._precision_transform = precision_transform
        self._settings = settings

    def to_domain(
        self, client_input: InterpolateModelClientInput
    ) -> InterpolateModel | None:
        base_model = self._find_model(client_input)
        if base_model is None:
            return None

        backend = base_model.backend
        if isinstance(backend, PyTorchBackend):
            backend = backend.model_copy(
                update={"accelerator": self._settings.torch_accelerator}
            )

        return InterpolateModel(
            id=base_model.id,
            variant=base_model.variant,
            interpolate_factor=client_input.interpolate_factor,
            file_path=base_model.file_path,
            description=base_model.description,
            url=base_model.url,
            precision=self._precision_transform.to_domain(
                self._settings.precision, base_model.backend.type
            ),
            backend=backend,
        )

    def _find_model(
        self, client_input: InterpolateModelClientInput
    ) -> InterpolateModelRepo | None:
        for model in self._model_repo.interpolate_models:
            if (
                model.id == client_input.id
                and model.backend.type == client_input.backed_type
            ):
                return model
        return None


class UpscaleModelTransformer:
    def __init__(
        self,
        model_repo: ModelRepo,
        precision_transform: PrecisionTransform,
        settings: PersistentSettingsProxy,
    ):
        self._model_repo = model_repo
        self._precision_transform = precision_transform
        self._settings = settings

    def to_domain(self, client_input: UpscaleModelClientInput) -> UpscaleModel | None:
        base_model = self._find_model(client_input)
        if base_model is None:
            return None

        backend = base_model.backend
        if isinstance(backend, PyTorchBackend):
            backend = backend.model_copy(
                update={"accelerator": self._settings.torch_accelerator}
            )

        return UpscaleModel(
            id=base_model.id,
            variant=base_model.variant,
            scale=base_model.scale,
            file_path=base_model.file_path,
            description=base_model.description,
            url=base_model.url,
            precision=self._precision_transform.to_domain(
                self._settings.precision, base_model.backend.type
            ),
            backend=backend,
        )

    def _find_model(
        self, client_input: UpscaleModelClientInput
    ) -> UpscaleModelRepo | None:
        for model in self._model_repo.upscale_models:
            if (
                model.id == client_input.id
                and model.backend.type == client_input.backed_type
            ):
                return model
        return None


class EnhancementModelTransformer:
    def __init__(
        self,
        model_repo: ModelRepo,
        precision_transform: PrecisionTransform,
        settings: PersistentSettingsProxy,
    ):
        self._model_repo = model_repo
        self._precision_transform = precision_transform
        self._settings = settings

    def to_domain(
        self, client_input: EnhancementModelClientInput
    ) -> EnhancementModel | None:
        base_model = self._find_model(client_input)
        if base_model is None:
            return None

        backend = base_model.backend
        if isinstance(backend, PyTorchBackend):
            backend = backend.model_copy(
                update={"accelerator": self._settings.torch_accelerator}
            )

        return EnhancementModel(
            id=base_model.id,
            variant=base_model.variant,
            file_path=base_model.file_path,
            description=base_model.description,
            url=base_model.url,
            precision=self._precision_transform.to_domain(
                self._settings.precision, base_model.backend.type
            ),
            backend=backend,
        )

    def _find_model(
        self, client_input: EnhancementModelClientInput
    ) -> EnhancementModelRepo | None:
        for model in self._model_repo.enhancement_models:
            if (
                model.id == client_input.id
                and model.backend.type == client_input.backed_type
            ):
                return model
        return None
