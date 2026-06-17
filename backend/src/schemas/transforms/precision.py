from __future__ import annotations

from src.schemas.domain.precision import NumpyPrecision, Precision, TorchPrecision


class PrecisionTransform:
    def to_domain(self, precision_id: str, backend_type: str) -> Precision:
        resolved = _resolve_auto(precision_id)
        if backend_type in ("pytorch", "tensorrt"):
            return TorchPrecision(precision_id=resolved)
        return NumpyPrecision(precision_id=resolved)


def _resolve_auto(precision_id: str) -> str:
    if precision_id == "auto":
        return "float32"
    return precision_id
