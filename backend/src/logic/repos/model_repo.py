from __future__ import annotations

import json
from pathlib import Path
from typing import List

import requests
from src.schemas.domain import (
    Backend,
    NCNNBackend,
    PyTorchBackend,
    TensorRTBackend,
)
from src.schemas.repo import (
    EnhancementModelRepo,
    InterpolateModelRepo,
    UpscaleModelRepo,
)
from src.schemas.repo.model import ModelRepoVariant

_REPO_DIR = Path(__file__).parent
_MODEL_REPO_JSON = _REPO_DIR / "model_repo.json"
_MODELS_DIR = _REPO_DIR / "models"
_MODEL_DOWNLOAD_BASE_URL = (
    "https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/"
)


class ModelRepo:
    """Loads model_repo.json into domain model instances."""

    def __init__(
        self,
        json_path: Path | None = None,
        backends: dict[str, dict] | None = None,
        models_dir: Path | None = None,
    ):
        self._json_path = json_path or _MODEL_REPO_JSON
        self._models_dir = models_dir or _MODELS_DIR
        self._backends = backends or {
            "ncnn": {"version": "0.0.0", "installed": False},
            "pytorch": {"version": "0.0.0", "installed": False, "accelerator": "CPU"},
            "tensorrt": {"version": "0.0.0", "installed": False},
        }

        self._models: List[ModelRepoVariant] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> List[ModelRepoVariant]:
        """Parse the JSON repo and populate internal model lists."""
        with open(self._json_path) as f:
            repo = json.load(f)

        self._models = []

        for backend_name, categories in repo.items():
            if backend_name == "onnx":
                continue

            backend_info = self._backends.get(
                backend_name, {"version": "0.0.0", "installed": False}
            )
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
    def models(self) -> List[ModelRepoVariant]:
        """All loaded models. Call .load() first."""
        return self._models

    @property
    def interpolate_models(self) -> List[InterpolateModelRepo]:
        return [m for m in self._models if isinstance(m, InterpolateModelRepo)]

    @property
    def upscale_models(self) -> List[UpscaleModelRepo]:
        return [m for m in self._models if isinstance(m, UpscaleModelRepo)]

    @property
    def enhancement_models(self) -> List[EnhancementModelRepo]:
        return [m for m in self._models if isinstance(m, EnhancementModelRepo)]

    def get_by_backend(self, backend_type: str) -> List[ModelRepoVariant]:
        """Filter models by backend type string (e.g. 'pytorch', 'ncnn')."""
        return [m for m in self._models if m.backend.type == backend_type]

    def ensure(self, model_id: str, backend_type: str) -> str:
        """Resolve a model's file path, downloading if needed.

        Returns the absolute path to the model file or directory.
        """
        for model in self._models:
            if model.id == model_id and model.backend.type == backend_type:
                self._ensure_model(model)
                return model.file_path
        raise ValueError(f"Model '{model_id}' not found for backend '{backend_type}'")

    # ------------------------------------------------------------------
    # Model file resolution
    # ------------------------------------------------------------------

    def _ensure_model(self, model: ModelRepoVariant) -> None:
        full_path = self._models_dir / model.file_path
        if not full_path.exists():
            url = _MODEL_DOWNLOAD_BASE_URL + model.file_path
            self._download(url, full_path)
        if full_path.suffix == ".gz" and full_path.name.endswith(".tar.gz"):
            extracted_dir = full_path.with_suffix("").with_suffix("")
            if not extracted_dir.exists():
                import tarfile

                with tarfile.open(full_path, "r:gz") as tar:
                    members = tar.getmembers()
                    prefixes = {m.name.split("/")[0] for m in members}
                    # Strip single top-level directory (e.g. rife-v4.6/) so contents
                    # land directly in extracted_dir instead of a nested wrapper.
                    if len(prefixes) == 1:
                        extracted_dir.mkdir(parents=True, exist_ok=True)
                        for member in members:
                            name = member.name
                            member.name = name.split("/", 1)[1] if "/" in name else name
                            tar.extract(member, path=extracted_dir)
                    else:
                        extracted_dir.mkdir(parents=True, exist_ok=True)
                        tar.extractall(path=extracted_dir)
                model.file_path = str(extracted_dir)
                full_path.unlink()
            else:
                model.file_path = str(extracted_dir)
                full_path.unlink()
        else:
            model.file_path = str(full_path)

    @staticmethod
    def _download(url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with dest.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_backend(
        backend_name: str, version: str, installed: bool, accelerator: str | None = None
    ) -> Backend:
        if backend_name == "ncnn":
            return NCNNBackend(version=version, installed=installed)
        if backend_name == "pytorch":
            return PyTorchBackend(
                version=version, installed=installed, accelerator=accelerator or "CPU"
            )
        if backend_name == "tensorrt":
            return TensorRTBackend(version=version, installed=installed)
        raise ValueError(f"Unknown backend: {backend_name}")

    @staticmethod
    def _parse_interpolate(data: dict, backend: Backend) -> InterpolateModelRepo:
        return InterpolateModelRepo(
            id=data["id"],
            variant=data["variant"],
            interpolate_factor=data.get("interpolate_factor", 2),
            file_path=data.get("file_path", ""),
            description=data.get("description"),
            url=data.get("url"),
            precision=data.get("precision", "float32"),
            backend=backend,
        )

    @staticmethod
    def _parse_upscale(data: dict, backend: Backend) -> UpscaleModelRepo:
        return UpscaleModelRepo(
            id=data["id"],
            variant=data["variant"],
            scale=data.get("scale", 2),
            file_path=data.get("file_path", ""),
            description=data.get("description"),
            url=data.get("url"),
            precision=data.get("precision", "float32"),
            backend=backend,
        )

    @staticmethod
    def _parse_enhancement(data: dict, backend: Backend) -> EnhancementModelRepo:
        return EnhancementModelRepo(
            id=data["id"],
            variant=data["variant"],
            file_path=data.get("file_path", ""),
            description=data.get("description"),
            url=data.get("url"),
            precision=data.get("precision", "float32"),
            backend=backend,
        )
