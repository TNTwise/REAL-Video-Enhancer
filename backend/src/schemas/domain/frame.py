from __future__ import annotations
from typing import Any, TYPE_CHECKING

import numpy as np

from src.utils.LogConfig import get_logger

if TYPE_CHECKING:
    from src.logic.render.backends.pytorch.TorchUtils import TorchUtils
from src.utils.Util import resize_image_np

logger = get_logger(__name__)


class Frame:
    def __init__(
        self,
        width: int,
        height: int,
        torch_utils: TorchUtils | None = None,
        hdr_mode: bool = False,
    ):
        self.width = width
        self.height = height
        self.hdr_mode = hdr_mode
        self._bit_depth = 3
        self.tensor_conversions = 0
        self._tensor: Any | None = None
        self._np: np.ndarray | None = None
        self._bytes: bytes | None = None
        self._torch_utils = torch_utils

    def _invalidate_cache(self, keep: str):
        if keep != "tensor":
            self._tensor = None
        if keep != "np":
            self._np = None
        if keep != "bytes":
            self._bytes = None

    def set_frame_bytes(self, frame: bytes) -> "Frame":
        if not isinstance(frame, bytes):
            raise TypeError(f"Expected bytes, got {type(frame).__name__}")
        self._invalidate_cache("bytes")
        self._bytes = frame
        return self

    def set_frame_tensor(self, frame: Any) -> "Frame":
        if self._torch_utils is not None and not isinstance(
            frame, self._torch_utils._torch.Tensor
        ):
            raise TypeError(f"Expected torch.Tensor, got {type(frame).__name__}")
        self._invalidate_cache("tensor")
        self._tensor = frame.clone()
        if self._torch_utils is not None:
            self._torch_utils.sync_all_streams()
        return self

    def set_frame_np(self, frame: Any) -> "Frame":
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(frame).__name__}")
        self._invalidate_cache("np")
        self._np = frame
        return self

    def get_frame_tensor(self, clear_cache: bool = False) -> Any:
        if self._tensor is None:
            if self._bytes is not None:
                if self._torch_utils is None:
                    raise RuntimeError("TorchUtils required to convert bytes to tensor")
                t = self._torch_utils._torch
                self._tensor = self.bytes_to_tensor(t.device("cpu"), t.float32)
            elif self._np is not None:
                if self._torch_utils is None:
                    raise RuntimeError("TorchUtils required to convert np to tensor")
                t = self._torch_utils._torch
                self._tensor = self._torch_utils.np_to_tensor(
                    self._np, t.device("cpu"), t.float32
                )
        if clear_cache:
            self._invalidate_cache("tensor")
        if self._tensor is None:
            raise RuntimeError("No frame data available")
        return self._tensor.clone()

    def get_frame_bytes(self, clear_cache: bool = False) -> bytes:
        if self._bytes is None:
            if self._tensor is not None:
                self._bytes = self.tensor_to_bytes(self._tensor)
            elif self._np is not None:
                self._bytes = self._np_to_bytes(self._np)
        if clear_cache:
            self._invalidate_cache("bytes")
        if self._bytes is None:
            raise RuntimeError("Bytes cannot be null")
        return self._bytes

    def get_frame_np(self, clear_cache: bool = False) -> Any:
        if self._np is None:
            if self._tensor is not None:
                if self._torch_utils is None:
                    raise RuntimeError("TorchUtils required to convert tensor to np")
                self._np = self._torch_utils.tensor_to_np(self._tensor)
            elif self._bytes is not None:
                self._np = self._bytes_to_np(self._bytes)
        if clear_cache:
            self._invalidate_cache("np")
        return self._np

    def bytes_to_tensor(self, device, dtype) -> Any:
        if self._torch_utils is None:
            raise RuntimeError("TorchUtils required for tensor conversion")
        t = self._torch_utils._torch
        data = self.get_frame_bytes()
        src_dtype = t.uint16 if self.hdr_mode else t.uint8
        frame = t.frombuffer(data, dtype=src_dtype)
        frame = (
            frame.to(device=device)
            .div(65535.0 if self.hdr_mode else 255.0)
            .clamp(0.0, 1.0)
            .reshape(self.height, self.width, 3)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .contiguous()
            .to(dtype=dtype)
        )
        return frame

    def tensor_to_bytes(self, tensor) -> bytes:
        if self._torch_utils is None:
            raise RuntimeError("TorchUtils required for tensor conversion")
        t = self._torch_utils._torch
        arr = (
            tensor.squeeze(0)
            .permute(1, 2, 0)
            .clamp(0.0, 1.0)
            .mul(65535.0 if self.hdr_mode else 255.0)
            .round()
            .to(t.uint16 if self.hdr_mode else t.uint8)
            .contiguous()
            .detach()
            .cpu()
        )
        return arr.numpy().tobytes()

    def _bytes_to_np(self, data: bytes) -> Any:
        channels = 3
        return np.frombuffer(
            data, dtype=np.uint8 if self._bit_depth == 8 else np.uint16
        ).reshape(self.height, self.width, int(channels))

    def _np_to_bytes(self, arr: Any) -> bytes:
        return arr.tobytes()

    def resize_frame(self, new_width: int, new_height: int) -> "Frame":
        if self._tensor is not None:
            if self._torch_utils is not None:
                self._tensor = self._torch_utils.resize_tensor(
                    self._tensor, new_width, new_height
                )
        if self._np is not None:
            self._np = resize_image_np(self._np, new_width, new_height)
        if self._bytes is not None:
            np_frame = self._bytes_to_np(self._bytes)
            resized_np = resize_image_np(np_frame, new_width, new_height)
            self._bytes = self._np_to_bytes(resized_np)
        self.width = new_width
        self.height = new_height
        return self

    def resize_frame_optimal(self, new_width: int, new_height: int):
        if self._tensor is not None:
            self._invalidate_cache("tensor")
        elif self._np is not None:
            self._invalidate_cache("np")
        elif self._bytes is not None:
            self._invalidate_cache("bytes")
        else:
            raise ValueError("Tried to resize nothing!")
        return self.resize_frame(new_width=new_width, new_height=new_height)

    def get_np_sdr(self):
        np_frame = self.get_frame_np()
        if self.hdr_mode:
            np_frame = (
                np.clip(np_frame.astype(np.float32) / 65535.0, 0, 1) * 255
            ).astype(np.uint8)
        return np_frame

    def clone(self) -> "Frame":
        new_frame = Frame(
            width=self.width,
            height=self.height,
            torch_utils=self._torch_utils,
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
            width=self.width,
            height=self.height,
            torch_utils=self._torch_utils,
            hdr_mode=self.hdr_mode,
        )
