import sys

import numpy as np

from src.logic.handlers.backend import TorchHandler
from src.logic.proxy import PersistentSettingsProxy
from src.schemas.domain.video_info import InputVideoInfo
from src.utils.LogConfig import get_logger

logger = get_logger(__name__)


def dummy_function(*args, **kwargs):
    """
    A dummy function that does nothing.
    This is used as a placeholder for device-specific functions that may not be available.
    """


def dummy_context_manager(*args, **kwargs):
    """
    A dummy context manager that does nothing.
    This is used as a placeholder for device-specific context managers that may not be available.
    """
    return DummyContextManager()


class DummyContextManager:
    def __call__(self, *args, **kwargs):
        """
        A dummy callable that returns a DummyContextManager instance.
        This is used as a placeholder for device-specific context managers that may not be available.
        """
        return self

    def __enter__(self):
        return self  # could return any resource

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.exception("An exception occurred")
        return False  # re-raise exceptions if any


class TorchUtils:
    """
    Torch Utils class. All general torch code is abstracted here.
    """

    # device and precision are in string formats, loaded straight from the command line arguments
    def __init__(
        self,
        input_video_info: InputVideoInfo,
        settings_proxy: PersistentSettingsProxy,
        torch_handler: TorchHandler,
    ):

        self._torch = torch_handler.get_torch()
        if self._torch is None:
            raise Exception("Torch does not exist")

        self.width = input_video_info.width
        self.height = input_video_info.height
        self.hdr_mode = settings_proxy.auto_hdr_mode and input_video_info.is_hdr
        self.gpu_id = settings_proxy.pytorch_gpu_id
        if settings_proxy.torch_accelerator_detection_mode == "auto":
            self.device_type = (
                "cuda"
                if self._self._torch.cuda.is_available()
                else "mps"
                if self._self._torch.backends.mps.is_available()
                else "xpu"
                if self._torch.xpu.is_available()
                else "cpu"
            )
        else:
            self.device_type = (
                "cuda"
                if "cu" in settings_proxy.torch_accelerator
                else "xpu"
                if "xpu" in settings_proxy.torch_accelerator
                else "mps"
                if "mps" in settings_proxy.torch_accelerator
                else "rocm"
                if "rocm" in settings_proxy.torch_accelerator
                else "cpu"
                if "cpu" in settings_proxy.torch_accelerator
                else None
            )
            if self.device_type is None:
                raise Exception("Invalid torch accelerator")
        try:
            test_tensor = self._torch.tensor([1.0]).cpu().numpy()
            del test_tensor
            self.use_numpy = True
        except Exception:
            logger.warning(
                "Failed to create a Numpy tensor; this will heavily reduce performance.",
                exc_info=True,
            )
            self.use_numpy = False
        self.__run_stream_func = self.__run_stream_function()
        self.__sync_all_streams_func = self.__sync_all_streams_function()
        self._torch_utils_stream = self.init_stream(
            gpu_id=int(settings_proxy.pytorch_gpu_id)
        )

    def __sync_all_streams_function(self):
        if self.device_type == "cuda":
            return self._torch.cuda.synchronize
        if self.device_type == "mps":
            return self._torch.mps.synchronize
        if self.device_type == "cpu":
            return dummy_function  # CPU does not require explicit synchronization
        if self.device_type == "xpu":
            return self._torch.xpu.synchronize
        return lambda: logger.warning(
            "Unknown device type %s, skipping stream synchronization.",
            self.device_type,
        )

    def init_stream(self, gpu_id=0):
        """
        Initializes the stream based on the device type.
        """
        logger.info(
            "Initializing stream for device %s (GPU ID: %s)",
            self.device_type,
            gpu_id,
        )
        device = self.handle_device(self.device_type, gpu_id)
        if self.device_type == "cuda":
            return self._torch.cuda.Stream(device=device)
        if self.device_type == "xpu":
            return self._torch.xpu.Stream(device=device)
        return DummyContextManager()  # For CPU and MPS, we can use a dummy stream

    def __run_stream_function(self) -> callable:
        """
        Runs the stream based on the device type.
        """
        if self.device_type == "cuda":
            return self._torch.cuda.stream
        if self.device_type == "xpu":
            return self._torch.xpu.stream
        return (
            dummy_context_manager  # For CPU and MPS, we can use a dummy context manager
        )

    def run_stream(self, stream):
        return self.__run_stream_func(stream)

    def sync_stream(self, stream: self._torch.Stream):
        match self.device_type:
            case "cuda" | "xpu":
                stream.synchronize()
            case "mps":
                self._torch.mps.synchronize()
            case "cpu":
                pass  # CPU does not require explicit synchronization
            case _:
                logger.warning(
                    "Unknown device type %s, skipping stream synchronization.",
                    self.device_type,
                )
                # For other devices, we assume no synchronization is needed.

    def sync_all_streams(self):
        """
        Synchronizes all streams based on the device type.
        """
        self.__sync_all_streams_func()

    @staticmethod
    def handle_device(device, gpu_id: int = 0):
        """
        Returns device based on gpu id and device parameter
        """
        logger.info("Handling device: %s, GPU ID: %s", device, gpu_id)
        if device == "auto":
            if self._torch.cuda.is_available():
                torchdevice = self._torch.device("cuda", gpu_id)
            else:
                torchdevice = self._torch.device(
                    "mps"
                    if self._torch.backends.mps.is_available()
                    else "xpu"
                    if self._torch.xpu.is_available()
                    else "cpu"
                )

        elif device == "cuda":
            torchdevice = self._torch.device(
                device, gpu_id
            )  # 0 is the device index, may have to change later
        else:
            torchdevice = self._torch.device(device)

        device = backendDetect.get_gpus_torch()[gpu_id]
        print("Using Device: " + str(device), file=sys.stderr)
        return torchdevice

    @staticmethod
    def handle_precision(precision) -> self._torch.dtype:
        logger.info("Handling precision: %s", precision)
        if precision == "auto":
            return (
                self._torch.float16
                if backendDetect.get_half_precision()
                else self._torch.float32
            )
        if precision == "float32":
            return self._torch.float32
        if precision == "float16":
            return self._torch.float16
        if precision == "bfloat16":
            return self._torch.bfloat16
        return self._torch.float32

    @self._torch.inference_mode()
    def copy_tensor(
        self,
        tensorToCopy: self._torch.Tensor,
        tensorCopiedTo: self._torch.Tensor,
        stream: self._torch.Stream,
    ):  # stream might be None
        """
        Docstring for copy_tensor

        :param self: Description
        :param tensorToCopy: Description
        :type tensorToCopy: self._torch.Tensor
        :param tensorCopiedTo: Description
        :type tensorCopiedTo: self._torch.Tensor
        :param stream: Description
        :type stream: self._torch.Stream
        """
        with self.run_stream(stream):  # type: ignore
            tensorToCopy.copy_(tensorCopiedTo, non_blocking=True)

            self.sync_stream(stream)

    def frame_to_tensor(
        self,
        frame,
    ):  # stream might be None
        """
        Docstring for frame_to_tensor

        :param frame: Frame data in bytes
        :param stream: Torch Stream for asynchronous operations
        :type stream: self._torch.Stream
        :param device: Target device for the tensor
        :type device: self._torch.device
        :param dtype: Target data type for the tensor
        :type dtype: self._torch.dtype
        :return: Tensor representation of the frame in the shape (1, C, H, W)
        :rtype: self._torch.Tensor
        """
        with self._torch.inference_mode():
            with self.run_stream(stream):  # type: ignore
                # ... (tensor creation and manipulation) ...
                frame = self._torch.frombuffer(
                    frame,
                    dtype=self._torch.uint16 if self.hdr_mode else self._torch.uint8,
                ).to(device=device, non_blocking=True)

                frame = (
                    frame.div(65535.0 if self.hdr_mode else 255.0)
                    .clamp(0.0, 1.0)
                    .reshape(self.height, self.width, 3)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .contiguous()
                ).to(dtype=dtype, non_blocking=True)

                self.sync_stream(stream)

            # No explicit sync for CPU here.
            return frame

    def clear_cache(self):
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
            self._torch.cuda.reset_max_memory_allocated()
            self._torch.cuda.reset_max_memory_cached()

    def tensor_to_frame(self, frame):
        """
        Docstring for tensor_to_frame

        :param self: Description
        :param frame: Description
        :type frame: self._torch.Tensor
        """
        with self._torch.inference_mode():
            # Prepare the tensor
            tensor = (
                frame.squeeze(0)
                .permute(1, 2, 0)
                .clamp(0.0, 1.0)
                .mul(65535.0 if self.hdr_mode else 255.0)
                .round()
                .to(self._torch.uint16 if self.hdr_mode else self._torch.uint8)
                .contiguous()
                .detach()
                .cpu()
            )
            if self.use_numpy:
                # Convert to numpy array if possible
                return tensor.numpy()
            np_dtype = np.uint16 if self.hdr_mode else np.uint8
            return np.array(tensor.tolist(), dtype=np_dtype)

    def np_to_tensor(self, arr: np.ndarray, device, dtype):
        """
        Docstring for np_to_tensor

        :param arr: Input numpy array in the shape (H, W, C)
        :type arr: ndarray
        :param device: Target device for the tensor
        :type device: self._torch.device
        :param dtype: Target data type for the tensor
        :type dtype: self._torch.dtype
        :return: Tensor representation of the numpy array in the shape (1, C, H
        , W)
        :rtype: self._torch.Tensor
        """
        return (
            self._torch.from_numpy(arr)
            .to(device=device, dtype=dtype)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

    def tensor_to_np(self, tensor) -> np.ndarray:
        """
        Docstring for tensor_to_np

        :param tensor: Input tensor in the shape (1, C, H, W)
        :type tensor: self._torch.Tensor
        :return: Numpy array representation of the tensor in the shape (H, W, C)
        :rtype: ndarray
        """
        return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    def resize_tensor(
        self,
        tensor,
        new_width: int,
        new_height: int,
        mode: str = "bilinear",
    ):
        """
        Docstring for resize_tensor

        :param tensor: Input tensor in the shape (1, C, H, W)
        :type tensor: self._torch.Tensor
        :param new_width: new width
        :type new_width: int
        :param new_height: new height
        :type new_height: int
        :param mode: Resizing mode, e.g., 'nearest', 'bilinear', 'bicubic'
        :type mode: str
        :return: Resized tensor in the shape (1, C, new_height, new_width)
        :rtype: torch.Tensor
        """

        return self._torch.nn.functional.interpolate(
            tensor,
            size=(new_height, new_width),
            mode=mode,
            align_corners=False
            if mode in ["linear", "bilinear", "bicubic", "trilinear"]
            else None,
        )
