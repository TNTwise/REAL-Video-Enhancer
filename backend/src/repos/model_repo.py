from __future__ import annotations

import json
from pathlib import Path
from typing import List

from src.schemas.domain import (
    InterpolateModel,
    UpscaleModel,
    EnhancementModel,
    ModelVariant,
    NCNNBackend,
    PyTorchBackend,
    TensorRTBackend,
    Backend,
)

_REPO_DIR = Path(__file__).parent
_MODEL_REPO_JSON = _REPO_DIR / "model_repo.json"


class ModelRepo:
    """Loads model_repo.json into domain model instances."""

    def __init__(
        self,
        json_path: Path | None = None,
        backends: dict[str, dict] | None = None,
    ):
        self._json_path = json_path or _MODEL_REPO_JSON
        self._backends = backends or {
            "ncnn": {"version": "0.0.0", "installed": False},
            "pytorch": {"version": "0.0.0", "installed": False, "accelerator": "CPU"},
            "tensorrt": {"version": "0.0.0", "installed": False},
        }

        self._models: List[ModelVariant] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> List[ModelVariant]:
        """Parse the JSON repo and populate internal model lists."""
        with open(self._json_path) as f:
            repo = json.load(f)

        self._models = []

        for backend_name, categories in repo.items():
            if backend_name == "onnx":
                continue

            backend_info = self._backends.get(backend_name, {"version": "0.0.0", "installed": False})
            backend = self._to_backend(
                backend_name,
                version=backend_info["version"],
                installed=backend_info["installed"],
                accelerator=backend_info.get("accelerator"),
            )

            for item in categories.get("interpolate", []):
                self._models.append(self._parse_interpolate(item, backend))

            for item in categories.get("upscale", []):
                self._models.append(self._parse_upscale(item, backend))

            for item in categories.get("enhancement", []):
                self._models.append(self._parse_enhancement(item, backend))

        self._loaded = True
        return self._models

    @property
    def models(self) -> List[ModelVariant]:
        """All loaded models. Call .load() first."""
        return self._models

    @property
    def interpolate_models(self) -> List[InterpolateModel]:
        return [m for m in self._models if isinstance(m, InterpolateModel)]

    @property
    def upscale_models(self) -> List[UpscaleModel]:
        return [m for m in self._models if isinstance(m, UpscaleModel)]

    @property
    def enhancement_models(self) -> List[EnhancementModel]:
        return [m for m in self._models if isinstance(m, EnhancementModel)]

    def get_by_backend(self, backend_type: str) -> List[ModelVariant]:
        """Filter models by backend type string (e.g. 'pytorch', 'ncnn')."""
        return [m for m in self._models if m.backend.type == backend_type]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_backend(backend_name: str, version: str, installed: bool, accelerator: str | None = None) -> Backend:
        if backend_name == "ncnn":
            return NCNNBackend(version=version, installed=installed)
        if backend_name == "pytorch":
            return PyTorchBackend(version=version, installed=installed, accelerator=accelerator or "CPU")
        if backend_name == "tensorrt":
            return TensorRTBackend(version=version, installed=installed)
        raise ValueError(f"Unknown backend: {backend_name}")

    @staticmethod
    def _parse_interpolate(data: dict, backend: Backend) -> InterpolateModel:
        return InterpolateModel(
            id=data["id"],
            variant=data["variant"],
            file_path=data.get("file_path", ""),
            description=data.get("description"),
            url=data.get("url"),
            backend=backend,
        )

    @staticmethod
    def _parse_upscale(data: dict, backend: Backend) -> UpscaleModel:
        return UpscaleModel(
            id=data["id"],
            variant=data["variant"],
            scale=data.get("scale", 2),
            file_path=data.get("file_path", ""),
            description=data.get("description"),
            url=data.get("url"),
            backend=backend,
        )

    @staticmethod
    def _parse_enhancement(data: dict, backend: Backend) -> EnhancementModel:
        return EnhancementModel(
            id=data["id"],
            variant=data["variant"],
            file_path=data.get("file_path", ""),
            description=data.get("description"),
            url=data.get("url"),
            backend=backend,
        )