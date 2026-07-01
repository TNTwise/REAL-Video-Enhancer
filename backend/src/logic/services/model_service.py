from __future__ import annotations

from typing import List

from src.logic.repos.model_repo import ModelRepo
from src.schemas.repo import (
    EnhancementModelRepo,
    InterpolateModelRepo,
    UpscaleModelRepo,
)


class ModelService:
    """Service for querying available models from the model repository."""

    def __init__(self, model_repo: ModelRepo | None = None):
        self._repo = model_repo or ModelRepo()
        if not self._repo._loaded:
            self._repo.load()

    def get_interpolate_models(
        self, backend_id: str | None = None
    ) -> List[InterpolateModelRepo]:
        models = self._repo.interpolate_models
        if backend_id:
            return [m for m in models if m.backend.type == backend_id]
        return models

    def get_upscale_models(
        self, backend_id: str | None = None
    ) -> List[UpscaleModelRepo]:
        models = self._repo.upscale_models
        if backend_id:
            return [m for m in models if m.backend.type == backend_id]
        return models

    def get_enhancement_models(
        self, backend_id: str | None = None
    ) -> List[EnhancementModelRepo]:
        models = self._repo.enhancement_models
        if backend_id:
            return [m for m in models if m.backend.type == backend_id]
        return models
