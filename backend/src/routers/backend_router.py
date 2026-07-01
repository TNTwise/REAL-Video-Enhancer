from fastapi import APIRouter
from src.logic.handlers.backend.ncnn_handler import NCNNHandler
from src.logic.handlers.backend.pytorch_handler import TorchHandler
from src.logic.handlers.backend.tensorrt_handler import TorchTensorRTHandler
from src.schemas.domain.backend import (
    AvailableBackends,
    NCNNBackend,
    PyTorchBackend,
    TensorRTBackend,
)

router = APIRouter(prefix="/backends", tags=["Backends"])


@router.get("/available", response_model=AvailableBackends)
def get_available_backends():
    torch = TorchHandler()
    ncnn = NCNNHandler()

    torch_version = torch.get_torch().__version__ if torch.is_available() else ""
    pytorch_device = (
        "cuda"
        if torch.is_available() and torch.get_torch().cuda.is_available()
        else "cpu"
    )

    return AvailableBackends(
        pytorch=PyTorchBackend(
            installed=torch.is_available(),
            version=torch_version,
            accelerator=pytorch_device,
        ),
        ncnn=NCNNBackend(
            installed=ncnn.is_available(),
            version="",
        ),
        tensorrt=TensorRTBackend(
            installed=TorchTensorRTHandler.is_available(),
            version="",
        ),
    )
