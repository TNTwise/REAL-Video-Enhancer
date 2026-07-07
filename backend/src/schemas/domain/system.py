from __future__ import annotations

from pydantic import BaseModel


class GPUInfo(BaseModel):
    name: str
    vendor: str | None = None
    memory_mb: int | None = None
    driver_version: str | None = None
    device_id: str | None = None


class SystemInfo(BaseModel):
    python_version: str
    app_version: str
    opencv_version: str | None = None
    pytorch_version: str | None = None
    cuda_version: str | None = None
    torch_accelerator: str = "cpu"
    total_memory_gb: float | None = None
    gpus: list[GPUInfo] = []
