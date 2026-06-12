from __future__ import annotations

from src.schemas.domain.precision import Precision
from pydantic import BaseModel, Field
from typing import Literal, Annotated, Union
from src.schemas.domain.backend import Backend
from src.schemas.request.model import InterpolateModelClientInput


class InterpolateModel(BaseModel):
    id: str
    type: Literal["interpolate"] = "interpolate"
    variant: str
    interpolate_factor: int
    file_path: str
    description: str | None = None
    url: str | None = None
    precision: Precision
    backend: Backend


class UpscaleModel(BaseModel):
    id: str
    type: Literal["upscale"] = "upscale"
    variant: str
    scale: int
    file_path: str
    precision: Precision
    description: str | None = None
    url: str | None = None
    backend: Backend


class EnhancementModel(BaseModel):
    type: Literal["enhancement"] = "enhancement"
    id: str
    variant: str
    file_path: str
    precision: Precision
    description: str | None = None
    url: str | None = None
    backend: Backend


ModelVariant = Annotated[
    Union[InterpolateModel, UpscaleModel, EnhancementModel], Field(discriminator="type")
]


class InterpolateModelTransformer:
    def __init__(self, model_repo: ModelRepo):
        self._model_repo = model_repo

    def to_domain(
        self, client_input: InterpolateModelClientInput
    ) -> InterpolateModel | None:
        base_model = self._find_model(client_input)
        if base_model is None:
            return None

        return InterpolateModel(
            id=base_model.id,
            variant=base_model.variant,
            interpolate_factor=client_input.interpolate_factor,
            file_path=base_model.file_path,
            description=base_model.description,
            url=base_model.url,
            precision=base_model.precision,
            backend=base_model.backend,
        )

    def _find_model(
        self, client_input: InterpolateModelClientInput
    ) -> InterpolateModel | None:
        for model in self._model_repo.interpolate_models:
            if (
                model.id == client_input.id
                and model.backend.type == client_input.backed_type
            ):
                return model
        return None
