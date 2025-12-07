import torch
import torch.nn.functional as F
import sys
from ..utils.BackendDetect import (
    BackendDetect
)
backendDetect = BackendDetect()

from ..utils.Util import (
    log,
    CudaChecker
)
HAS_PYTORCH_CUDA = CudaChecker().HAS_PYTORCH_CUDA
import numpy as np

def dummy_function(*args, **kwargs):
    """
    A dummy function that does nothing.
    This is used as a placeholder for device-specific functions that may not be available.
    """
    pass

def dummy_context_manager(*args, **kwargs):
    """
    A dummy context manager that does nothing.
    This is used as a placeholder for device-specific context managers that may not be available.
    """
    return DummyContextManager()

class DummyContextManager:
    def __call__ (self, *args, **kwargs):
        """
        A dummy callable that returns a DummyContextManager instance.
        This is used as a placeholder for device-specific context managers that may not be available.
        """
        return self
    def __enter__(self):
        return self  # could return any resource

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            print(f"An exception occurred: {exc_type}")
        return False  # re-raise exceptions if any


class TorchUtils:
    # device and precision are in string formats, loaded straight from the command line arguments
    def __init__(self, width, height, device_type:str, hdr_mode=False, padding=None, gpu_id=0):
        self.width = width
        self.height = height
        self.hdr_mode = hdr_mode
        self.gpu_id = gpu_id
        if device_type == "auto":
            self.device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "xpu" if torch.xpu.is_available() else "cpu"    
        else:
            self.device_type = device_type
        try:
            test_tensor = torch.tensor([1.0]).cpu().numpy()
            del test_tensor
            self.use_numpy = True
        except Exception as e:
            log(f"Failed to create a Numpy tensor, This will heavily reduce performance.")
            self.use_numpy = False
        self.__run_stream_func = self.__run_stream_function()
        self.__sync_all_streams_func = self.__sync_all_streams_function()

        # persistent pinned buffer for async GPU uploads
        self._pinned_buffer = None
        self._pinned_buffer_numel = 0
        self._pinned_buffer_dtype = None

    def __sync_all_streams_function(self):
        if self.device_type == "cuda":
            return torch.cuda.synchronize
        if self.device_type == "mps":
            return torch.mps.synchronize
        if self.device_type == "cpu":
            return dummy_function  # CPU does not require explicit synchronization
        if self.device_type == "xpu":
            return torch.xpu.synchronize
        return lambda: log(f"Unknown device type {self.device_type}, skipping stream synchronization.")

    def init_stream(self, gpu_id = 0) -> torch.Stream:
        """
        Initializes the stream based on the device type.
        """
        log(f"Initializing stream for device {self.device_type} (GPU ID: {gpu_id})")
        device = self.handle_device(self.device_type, gpu_id)
        if self.device_type == "cuda":
            return torch.cuda.Stream(device=device)
        elif self.device_type == "xpu":
            return torch.xpu.Stream(device=device)
        else:
            return DummyContextManager()  # For CPU and MPS, we can use a dummy stream

    def __run_stream_function(self) -> callable:
        """
        Runs the stream based on the device type.
        """
        if self.device_type == "cuda":
            return torch.cuda.stream
        elif self.device_type == "xpu":
            return torch.xpu.stream
        else:
            return  dummy_context_manager # For CPU and MPS, we can use a dummy context manager


    def run_stream(self, stream):
        return self.__run_stream_func(stream) 

    def sync_stream(self, stream: torch.Stream):
        match self.device_type:
            case "cuda" | "xpu":
                stream.synchronize()
            case "mps":
                torch.mps.synchronize()
            case "cpu":
                pass  # CPU does not require explicit synchronization
            case _:
                log(f"Unknown device type {self.device_type}, skipping stream synchronization.")
                # For other devices, we assume no synchronization is needed.
        
    def sync_all_streams(self):
        """
        Synchronizes all streams based on the device type.
        """
        self.__sync_all_streams_func()
             
    @staticmethod
    def handle_device(device, gpu_id: int = 0) -> torch.device:
        """
        returns device based on gpu id and device parameter
        """
        log(f"Handling device: {device}, GPU ID: {gpu_id}")
        if device == "auto":
            if torch.cuda.is_available():
                torchdevice = torch.device("cuda", gpu_id)
            else:
                torchdevice = torch.device("mps" if torch.backends.mps.is_available() else "xpu" if torch.xpu.is_available() else "cpu")
        
        elif device == "cuda":
            torchdevice = torch.device(
                device, gpu_id
            )  # 0 is the device index, may have to change later
        else:
            torchdevice = torch.device(device)
    
        device = backendDetect.get_gpus_torch()[gpu_id]
        print("Using Device: " + str(device), file=sys.stderr)
        return torchdevice

    @staticmethod
    def handle_precision(precision) -> torch.dtype:
        log(f"Handling precision: {precision}")
        if precision == "auto":
            return torch.float16 if backendDetect.get_half_precision() else torch.float32
        if precision == "float32":
            return torch.float32
        if precision == "float16":
            return torch.float16
        if precision == "bfloat16":
            return torch.bfloat16
        return torch.float32

    @torch.inference_mode()
    def copy_tensor(self, tensorToCopy: torch.Tensor, tensorCopiedTo: torch.Tensor, stream: torch.Stream): # stream might be None
        with self.run_stream(stream):  # type: ignore
            tensorToCopy.copy_(tensorCopiedTo, non_blocking=True)
        
            self.sync_stream(stream)

    @torch.inference_mode()
    def frame_to_tensor(self, frame, stream: torch.Stream, device: torch.device, dtype: torch.dtype) -> torch.Tensor: # stream might be None
        with self.run_stream(stream):  # type: ignore
             # ... (tensor creation and manipulation) ...
            frame = torch.frombuffer(
                    frame,
                    dtype=torch.uint16 if self.hdr_mode else torch.uint8,
                ).to(device=device, non_blocking=True) 
            
            frame = (
                frame
                .div(65535.0 if self.hdr_mode else 255.0)
                .clamp(0.0, 1.0)
                .reshape(self.height, self.width, 3)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .contiguous()
                ).to(dtype=dtype, non_blocking=True)
                
            self.sync_stream(stream)

        # No explicit sync for CPU here.
        return frame
    
    @staticmethod
    def clear_cache():
        if HAS_PYTORCH_CUDA:
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_max_memory_cached()
    
    @torch.inference_mode()
    def tensor_to_frame(self, frame: torch.Tensor):
        # Prepare the tensor
        tensor = (
            frame.squeeze(0)
            .permute(1, 2, 0)
            .clamp(0., 1.)
            .mul(65535.0 if self.hdr_mode else 255.0)
            .round()
            .to(torch.uint16 if self.hdr_mode else torch.uint8)
            .contiguous()
            .detach()
            .cpu()
        )
        if self.use_numpy:
            # Convert to numpy array if possible
            return tensor.numpy()
        else:
            np_dtype = np.uint16 if self.hdr_mode else np.uint8
            return np.array(tensor.tolist(), dtype=np_dtype)
    
    @staticmethod
    @torch.inference_mode()
    def np_to_tensor(arr: np.ndarray, device, dtype):
        import torch
        return torch.from_numpy(arr).to(device=device, dtype=dtype).permute(2, 0, 1).unsqueeze(0)

    
    @staticmethod
    @torch.inference_mode()
    def tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
        return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    @staticmethod
    @torch.inference_mode()
    def resize_tensor(
        tensor: torch.Tensor,
        new_width: int,
        new_height: int,
        mode: str = "bilinear",
    ) -> torch.Tensor:
        return F.interpolate(
            tensor,
            size=(new_height, new_width),
            mode=mode,
            align_corners=False if mode in ["linear", "bilinear", "bicubic", "trilinear"] else None,
        )