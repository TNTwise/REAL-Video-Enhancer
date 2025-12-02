from typing import Any, Optional
import numpy as np

_pytorch_device = None
_pytorch_dtype = None
_pytorch_steam = None
_torch_utils = None
_torch = None

def _init_pytorch(device, gpu_id, dtype, width, height, hdr_mode):
    global _pytorch_device, _pytorch_dtype, _torch_utils, _torch, _pytorch_steam
    if _torch_utils is None:
        import torch
        from ..pytorch.TorchUtils import TorchUtils
        _torch = torch
        _torch_utils = TorchUtils(
            width=width, 
            height=height, 
            device_type=device, 
            hdr_mode=hdr_mode, 
            gpu_id=gpu_id
            )
        _pytorch_steam = _torch_utils.init_stream(gpu_id=gpu_id)
        _pytorch_device = _torch_utils.handle_device(device, gpu_id)
        _pytorch_dtype = _torch_utils.handle_precision(dtype)


class Frame:
    def __init__(self, backend: str, width: int, height: int, device, gpu_id, hdr_mode, dtype):
        self.width = width
        self.height = height
        self.gpu_id = gpu_id
        self.device = device
        self.hdr_mode = hdr_mode
        self.dtype = dtype
        
        self._tensor: Optional[Any] = None
        self._np: Optional[np.ndarray] = None
        self._bytes: Optional[bytes] = None
        
        if backend in ("pytorch", "tensorrt"):
            _init_pytorch(device, gpu_id, dtype, width, height, hdr_mode)

    def _invalidate_cache(self, keep: str):
        """Clear cached representations except the one being set."""
        if keep != "tensor":
            self._tensor = None
        if keep != "np":
            self._np = None
        if keep != "bytes":
            self._bytes = None

    def set_frame_bytes(self, frame: bytes):
        if not isinstance(frame, bytes):
            raise TypeError(f"Expected bytes, got {type(frame).__name__}")
        self._invalidate_cache("bytes")
        self._bytes = frame

    def set_frame_tensor(self, frame: Any):
        if _torch is not None and not isinstance(frame, _torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(frame).__name__}")
        self._invalidate_cache("tensor")
        self._tensor = frame

    def set_frame_np(self, frame: np.ndarray):
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(frame).__name__}")
        self._invalidate_cache("np")
        self._np = frame

    # --- Lazy Getters ---
    def get_frame_tensor(self) -> Any:
        if self._tensor is None:
            if self._bytes is not None:
                self._tensor = _torch_utils.frame_to_tensor(
                    self._bytes, _pytorch_steam, _pytorch_device, _pytorch_dtype, 
                )
            elif self._np is not None:
                self._tensor = _torch_utils.np_to_tensor(
                    self._np, _pytorch_device, _pytorch_dtype
                )
        return self._tensor

    def get_frame_bytes(self) -> bytes:
        if self._bytes is None:
            if self._tensor is not None:
                self._bytes = _torch_utils.tensor_to_frame(self._tensor)
            elif self._np is not None:
                self._bytes = self._np_to_bytes(self._np)
        return self._bytes

    def get_frame_np(self) -> np.ndarray:
        if self._np is None:
            if self._tensor is not None:
                self._np = _torch_utils.tensor_to_np(self._tensor)
            elif self._bytes is not None:
                self._np = self._bytes_to_np(self._bytes)
        return self._np

    # --- Conversion helpers ---
    def _bytes_to_np(self, data: bytes) -> np.ndarray:
        # Assuming raw RGB/BGR bytes
        channels = 6 if self.hdr_mode else 3
        return np.frombuffer(data, dtype=np.uint8 if not self.hdr_mode else np.uint16).reshape(
            self.height, self.width, channels
        )

    def _np_to_bytes(self, arr: np.ndarray) -> bytes:
        return arr.tobytes()