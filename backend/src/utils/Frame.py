from typing import Any
from ..utils.Util import log
global pytorch_device, pytorch_dtype 
pytorch_device = None
pytorch_dtype = None
from ..pytorch.TorchUtils import TorchUtils
class Frame:
    
    def __init__(self, backend: str, width: int, height: int, device, gpu_id, hdr_mode, dtype):
        self.width = width
        self.height = height
        self.gpu_id = gpu_id
        self.device = device
        self.hdr_mode = hdr_mode
        self.dtype = dtype
        self.frame_tensor = None
        self.frame_bytes = None
        self.frame_type = None
        global pytorch_device, pytorch_dtype
        if (backend == "pytorch" or backend == "tensorrt") and pytorch_device is None:
            
            pytorch_device = TorchUtils.handle_device(device, gpu_id)
            pytorch_dtype = TorchUtils.handle_precision(dtype)
    
    def set_frame_bytes(self, frame: Any):
        self.frame_type = type(frame)
        if self.frame_type == bytes:
            self.frame_bytes = frame
        else:
            self.frame_bytes = TorchUtils.tensor_to_frame(frame, self.hdr_mode)

    def set_frame_tensor(self, frame: Any):
        self.frame_type = type(frame)
        if self.frame_type != bytes:
            self.frame_tensor = frame
        else:
            self.frame_tensor = TorchUtils.frame_to_tensor(self.frame_bytes, pytorch_device, pytorch_dtype, self.hdr_mode, self.width, self.height)

    def get_frame_tensor(self) -> Any:
        if self.frame_tensor is not None:
            return self.frame_tensor
        else:
            self.frame_tensor = TorchUtils.frame_to_tensor(self.frame_bytes, pytorch_device, pytorch_dtype, self.hdr_mode, self.width, self.height)
            return self.frame_tensor
    
    def get_frame_bytes(self) -> bytes:
        if self.frame_bytes is not None:
            return self.frame_bytes
        else:
            self.frame_bytes = TorchUtils.tensor_to_frame(self.frame_tensor, self.hdr_mode)
            return self.frame_bytes