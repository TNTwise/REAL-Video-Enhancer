import os
import sys

import pytest


BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BACKEND_DIR not in sys.path:
	sys.path.insert(0, BACKEND_DIR)


@pytest.fixture()
def frame_module():
	# Import as a module so we can monkeypatch module-level globals.
	import src.utils.Frame as frame_module

	return frame_module


def _make_sdr_np(width: int, height: int):
	np = pytest.importorskip("numpy")
	return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


def _make_hdr_np(width: int, height: int):
	np = pytest.importorskip("numpy")
	return np.random.randint(0, 65536, (height, width, 3), dtype=np.uint16)


def test_set_frame_bytes_requires_bytes(frame_module):
	Frame = frame_module.Frame
	f = Frame(backend="onnx", width=2, height=2, device="cpu", gpu_id=0, hdr_mode=False, dtype="float32")

	with pytest.raises(TypeError):
		f.set_frame_bytes("not-bytes")  # type: ignore[arg-type]


def test_bytes_to_np_roundtrip_sdr(frame_module):
	np = pytest.importorskip("numpy")
	Frame = frame_module.Frame

	width, height = 4, 3
	arr = _make_sdr_np(width, height)
	raw = arr.tobytes()

	f = Frame(backend="onnx", width=width, height=height, device="cpu", gpu_id=0, hdr_mode=False, dtype="float32")
	f.set_frame_bytes(raw)

	arr2 = f.get_frame_np()
	assert isinstance(arr2, np.ndarray)
	assert arr2.shape == (height, width, 3)
	assert arr2.dtype == np.uint8
	assert np.array_equal(arr2, arr)
	assert f.get_frame_bytes() == raw


def test_bytes_to_np_roundtrip_hdr(frame_module):
	np = pytest.importorskip("numpy")
	Frame = frame_module.Frame

	width, height = 4, 3
	arr = _make_hdr_np(width, height)
	raw = arr.tobytes()

	f = Frame(backend="onnx", width=width, height=height, device="cpu", gpu_id=0, hdr_mode=True, dtype="float32")
	f.set_frame_bytes(raw)

	arr2 = f.get_frame_np()
	assert isinstance(arr2, np.ndarray)
	assert arr2.shape == (height, width, 3)
	assert arr2.dtype == np.uint16
	assert np.array_equal(arr2, arr)
	assert f.get_frame_bytes() == raw


def test_get_np_sdr_converts_from_hdr(frame_module):
	np = pytest.importorskip("numpy")
	Frame = frame_module.Frame

	width, height = 2, 2
	arr = np.array(
		[
			[[0, 0, 0], [65535, 65535, 65535]],
			[[32768, 32768, 32768], [1, 2, 3]],
		],
		dtype=np.uint16,
	)
	f = Frame(backend="onnx", width=width, height=height, device="cpu", gpu_id=0, hdr_mode=True, dtype="float32")
	f.set_frame_np(arr)

	sdr = f.get_np_sdr()
	assert sdr.dtype == np.uint8
	assert sdr.shape == (height, width, 3)
	assert int(sdr[0, 0, 0]) == 0
	assert int(sdr[0, 1, 0]) == 255


def test_clone_is_deep_copy_for_np(frame_module):
	np = pytest.importorskip("numpy")
	Frame = frame_module.Frame

	width, height = 3, 3
	arr = _make_sdr_np(width, height)
	f = Frame(backend="onnx", width=width, height=height, device="cpu", gpu_id=0, hdr_mode=False, dtype="float32")
	f.set_frame_np(arr)

	g = f.clone()
	g_np = g.get_frame_np()
	assert np.array_equal(g_np, arr)

	g_np[0, 0, 0] = (g_np[0, 0, 0] + 1) % 255
	assert not np.array_equal(f.get_frame_np(), g_np)


def test_resize_frame_updates_size_and_data(frame_module):
	cv2 = pytest.importorskip("cv2")
	np = pytest.importorskip("numpy")
	Frame = frame_module.Frame

	width, height = 8, 6
	arr = _make_sdr_np(width, height)

	f = Frame(backend="onnx", width=width, height=height, device="cpu", gpu_id=0, hdr_mode=False, dtype="float32")
	f.set_frame_np(arr)
	f.resize_frame(new_width=4, new_height=3)
	assert f.width == 4
	assert f.height == 3

	arr2 = f.get_frame_np()
	assert isinstance(arr2, np.ndarray)
	assert arr2.shape == (3, 4, 3)


class MockTorchUtilsCPU:
	"""Minimal CPU-only TorchUtils mock for testing Frame's torch paths."""

	def __init__(self, width: int, height: int, hdr_mode: bool = False):
		self.width = width
		self.height = height
		self.hdr_mode = hdr_mode

	def init_stream(self, gpu_id: int = 0):
		return None

	@staticmethod
	def handle_device(device, gpu_id: int = 0):
		import torch

		return torch.device("cpu")

	@staticmethod
	def handle_precision(precision):
		import torch

		if precision in ("float16", "fp16"):
			return torch.float16
		if precision in ("bfloat16", "bf16"):
			return torch.bfloat16
		return torch.float32

	@staticmethod
	def sync_all_streams():
		return None

	def frame_to_tensor(self, frame: bytes, stream, device, dtype):
		import torch
		np = pytest.importorskip("numpy")

		np_dtype = np.uint16 if self.hdr_mode else np.uint8
		arr = np.frombuffer(frame, dtype=np_dtype).reshape(self.height, self.width, 3)
		t = torch.from_numpy(arr)
		t = t.to(device=device)
		t = t.to(dtype=torch.float32)
		t = t.div(65535.0 if self.hdr_mode else 255.0).clamp(0.0, 1.0)
		t = t.permute(2, 0, 1).unsqueeze(0).contiguous()
		return t.to(dtype=dtype)

	@staticmethod
	def np_to_tensor(arr, device, dtype):
		import torch

		return torch.from_numpy(arr).to(device=device, dtype=dtype).permute(2, 0, 1).unsqueeze(0)

	@staticmethod
	def tensor_to_np(tensor):
		np = pytest.importorskip("numpy")

		return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

	@staticmethod
	def resize_tensor(tensor, new_width: int, new_height: int):
		import torch.nn.functional as F

		return F.interpolate(tensor, size=(new_height, new_width), mode="bilinear", align_corners=False)


def test_get_frame_tensor_from_bytes_uses_cpu_mock(frame_module, monkeypatch):
	torch = pytest.importorskip("torch")
	np = pytest.importorskip("numpy")
	Frame = frame_module.Frame

	width, height = 4, 3
	arr = _make_sdr_np(width, height)
	raw = arr.tobytes()

	def _init_pytorch_stub(device, gpu_id, dtype, width, height, hdr_mode):
		frame_module._torch = torch
		frame_module._torch_utils = MockTorchUtilsCPU(width=width, height=height, hdr_mode=hdr_mode)
		frame_module._pytorch_stream = None
		frame_module._pytorch_device = torch.device("cpu")
		frame_module._pytorch_dtype = torch.float32

	monkeypatch.setattr(frame_module, "_init_pytorch", _init_pytorch_stub)

	f = Frame(backend="pytorch", width=width, height=height, device="cuda", gpu_id=0, hdr_mode=False, dtype="float32")
	f.set_frame_bytes(raw)
	t = f.get_frame_tensor()

	assert isinstance(t, torch.Tensor)
	assert t.device.type == "cpu"
	assert t.shape == (1, 3, height, width)
	assert t.dtype == torch.float32
	# Spot-check the conversion is in range [0, 1].
	assert float(t.min()) >= 0.0
	assert float(t.max()) <= 1.0

	# Ensure returned tensor is a clone (mutating it doesn't change cached tensor).
	t2 = f.get_frame_tensor()
	t2.zero_()
	t3 = f.get_frame_tensor()
	assert float(t3.max()) > 0.0


def test_set_frame_tensor_type_check_with_mock(frame_module, monkeypatch):
	torch = pytest.importorskip("torch")
	Frame = frame_module.Frame

	def _init_pytorch_stub(device, gpu_id, dtype, width, height, hdr_mode):
		frame_module._torch = torch
		frame_module._torch_utils = MockTorchUtilsCPU(width=width, height=height, hdr_mode=hdr_mode)
		frame_module._pytorch_stream = None
		frame_module._pytorch_device = torch.device("cpu")
		frame_module._pytorch_dtype = torch.float32

	monkeypatch.setattr(frame_module, "_init_pytorch", _init_pytorch_stub)

	f = Frame(backend="pytorch", width=2, height=2, device="auto", gpu_id=0, hdr_mode=False, dtype="float32")

	with pytest.raises(TypeError):
		f.set_frame_tensor("not-a-tensor")  # type: ignore[arg-type]


