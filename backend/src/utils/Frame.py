from typing import Any, Optional
try:
    import numpy as np
except ImportError:
    pass
from .Util import resize_image_np, log
_pytorch_device = None
_pytorch_dtype = None
_pytorch_stream = None
_torch_utils = None
_torch = None

def _init_pytorch(device, gpu_id, dtype, width, height, hdr_mode):
    global _pytorch_device, _pytorch_dtype, _torch_utils, _torch, _pytorch_stream
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
        _pytorch_stream = _torch_utils.init_stream(gpu_id=gpu_id)
        _pytorch_device = _torch_utils.handle_device(device, gpu_id)
        _pytorch_dtype = _torch_utils.handle_precision(dtype)
        print("Initialized Frame PyTorch utils")


class Frame:
    def __init__(self, backend: str, width: int, height: int, device, gpu_id, hdr_mode, dtype):
        self.backend = backend
        self.width = width
        self.height = height
        self.gpu_id = gpu_id
        self.device = device
        self.hdr_mode = hdr_mode
        self.dtype = dtype
        
        self.tensor_conversions = 0
        self._tensor: Optional[Any] = None
        self._np: Optional[np.ndarray] = None
        self._bytes: Optional[bytes] = None
        
        if backend in ("pytorch", "tensorrt"):
            _init_pytorch(device, gpu_id, dtype, width, height, hdr_mode)

    def _invalidate_cache(self, keep: str):
        """Clear cached representations except the one being set."""
        if keep != "tensor":
            del self._tensor
            self._tensor = None
        if keep != "np":
            del self._np
            self._np = None
        if keep != "bytes":
            del self._bytes
            self._bytes = None

    def set_frame_bytes(self, frame: bytes) -> "Frame":
        if not isinstance(frame, bytes):
            raise TypeError(f"Expected bytes, got {type(frame).__name__}")
        self._invalidate_cache("bytes")
        self._bytes = frame
        return self
    
    def set_frame_tensor(self, frame: Any) -> "Frame":
        # might need to sync streams here
        if _torch is not None and not isinstance(frame, _torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(frame).__name__}")
        self._invalidate_cache("tensor")
        self._tensor = frame.clone()
        _torch_utils.sync_all_streams()
        return self

    def set_frame_np(self, frame: Any) -> "Frame":
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(frame).__name__}")
        self._invalidate_cache("np")
        self._np = frame
        return self
    # --- Lazy Getters ---
    
    def get_frame_tensor(self, clear_cache:bool =False) -> Any:
        """
        Get the frame as a torch tensor in format (1, C, H, W).
        """
        
        if self._tensor is None:
            if self._bytes is not None:
                self._tensor = _torch_utils.frame_to_tensor(
                    self._bytes, _pytorch_stream, _pytorch_device, _pytorch_dtype, 
                )
            elif self._np is not None:
                self._tensor = _torch_utils.np_to_tensor(
                    self._np, _pytorch_device, _pytorch_dtype
                )
        if clear_cache:
            self._invalidate_cache("tensor")
            
        return self._tensor.clone() # this helps so the frame wont be overwritten? have to test later.

    def get_frame_bytes(self, clear_cache: bool = False) -> bytes:
        if self._bytes is None:
            if self._tensor is not None:
                self._bytes = _torch_utils.tensor_to_frame(self._tensor)
            elif self._np is not None:
                self._bytes = self._np_to_bytes(self._np)

        if clear_cache:
            self._invalidate_cache("bytes")

        return self._bytes

    def get_frame_np(self, clear_cache: bool = False) -> Any:
        """
        Get the frame as a numpy array in format (H, W, C).
        """
        if self._np is None:
            if self._tensor is not None:
                self._np = _torch_utils.tensor_to_np(self._tensor)
                
            elif self._bytes is not None:
                self._np = self._bytes_to_np(self._bytes)

        if clear_cache:
            self._invalidate_cache("np")
        return self._np

    # --- Conversion helpers ---
    def _bytes_to_np(self, data: bytes) -> Any:
        # Assuming raw RGB/BGR bytes
        channels = 3
        
        return np.frombuffer(data, dtype=np.uint8 if not self.hdr_mode else np.uint16).reshape(
            self.height, self.width, int(channels)
        )

    def _np_to_bytes(self, arr: Any) -> bytes:
        return arr.tobytes()

    def resize_frame(self, new_width: int, new_height: int) -> "Frame":
        
        if self._tensor is not None:
            self._tensor = _torch_utils.resize_tensor(
                self._tensor, new_width, new_height
            )
        if self._np is not None:
            self._np = resize_image_np(
                self._np, new_width, new_height
            )
        if self._bytes is not None:
            np_frame = self._bytes_to_np(self._bytes)
            resized_np = resize_image_np(
                np_frame, new_width, new_height
            )
            self._bytes = self._np_to_bytes(resized_np)
        self.width = new_width
        self.height = new_height
        return self
    
    def get_np_sdr(self):
        """
        Get the frame as a numpy array in SDR format (H, W, C) with dtype uint8.
        """
        np_frame = self.get_frame_np()
        if self.hdr_mode:
            # Convert from HDR (uint16) to SDR (uint8)
            np_frame = (np.clip(np_frame.astype(np.float32) / 65535.0, 0, 1) * 255).astype(np.uint8)
        return np_frame

    
    def clone(self) -> "Frame":
        new_frame = Frame(
            backend=self.backend,
            width=self.width,
            height=self.height,
            device=self.device,
            gpu_id=self.gpu_id,
            hdr_mode=self.hdr_mode,
            dtype=self.dtype,
        )
        if self._tensor is not None:
            new_frame.set_frame_tensor(self._tensor.clone())
        if self._np is not None:
            new_frame.set_frame_np(self._np.copy())
        if self._bytes is not None:
            new_frame.set_frame_bytes(self._bytes)
        return new_frame
    
    def get_dummy_frame(self) -> "Frame":
        return Frame(
            backend=self.backend,
            width=self.width,
            height=self.height,
            device=self.device,
            gpu_id=self.gpu_id,
            hdr_mode=self.hdr_mode,
            dtype=self.dtype,
        )
