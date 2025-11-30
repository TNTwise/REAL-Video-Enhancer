from typing import Any
from ..utils.Util import log
class Frame:
    
    def __init__(self, backend: str, width: int, height: int, device, gpu_id, hdr_mode, dtype):
        self.width = width
        self.height = height
        self.gpu_id = gpu_id
        self.device = device
        self.hdr_mode = hdr_mode
        self.dtype = dtype
        if backend == "pytorch" or backend == "tensorrt":
            from ..pytorch.TorchUtils import TorchUtils
            self.torch_utils = TorchUtils
            self.pytorch_device = TorchUtils.handle_device(device, gpu_id)
            self.pytorch_dtype = TorchUtils.handle_precision(dtype)
    
    def set_frame_bytes(self, frame: Any):
        self.frame_type = type(frame)
        if self.frame_type == bytes:
            self.frame_bytes = frame
        else:
            self.frame_bytes = self.torch_utils.tensor_to_frame(frame, self.hdr_mode)

    def set_frame_tensor(self, frame: Any):
        self.frame_type = type(frame)
        if self.frame_type != bytes:
            self.frame_tensor = frame
        else:
            self.frame_tensor = self.torch_utils.frame_to_tensor(frame, self.pytorch_device, self.pytorch_dtype)

    def get_frame_tensor(self) -> Any:
        if self.frame_tensor:
            return self.frame_tensor
        else:
            log("WARN: Converting frame bytes to tensor on the fly!")
            self.frame_tensor = self.torch_utils.frame_to_tensor(self.frame_bytes, self.pytorch_device, self.pytorch_dtype)
            return self.frame_tensor
    
    def get_frame_bytes(self) -> bytes:
        if self.frame_bytes:
            return self.frame_bytes
        else:
            log("WARN: Converting frame tensor to bytes on the fly!")
            self.frame_bytes = self.torch_utils.tensor_to_frame(self.frame_tensor)
            return self.frame_bytes