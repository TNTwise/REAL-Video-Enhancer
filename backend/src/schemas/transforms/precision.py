from __future__ import annotations

from src.schemas.domain.precision import NumpyPrecision, Precision, TorchPrecision


class PrecisionTransform:
    def to_domain(self, precision_id: str, backend_type: str) -> Precision:
        if backend_type in ("pytorch", "tensorrt"):
            return TorchPrecision(precision_id=precision_id)
        return NumpyPrecision(precision_id=precision_id)
