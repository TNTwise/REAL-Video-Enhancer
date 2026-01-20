from ...models.pytorch_msssim import ssim_matlab
from torch.nn import functional as F
from PIL import ImageDraw, ImageFont
from torchvision import transforms
import cv2
import numpy as np
import torch
import math
from queue import Queue
import _thread
import subprocess
import logging


logger = logging.getLogger(__name__)


def check_cupy_env():
    SUPPORT_CUPY = True
    try:
        import cupy

        if cupy.cuda.get_cuda_path() == None:
            SUPPORT_CUPY = False
    except ImportError:
        SUPPORT_CUPY = False
    except Exception:
        logger.exception("Error while checking CuPy environment")
        SUPPORT_CUPY = False

    return SUPPORT_CUPY


def check_scene(x1, x2, scdet_threshold=0.3):
    x1 = F.interpolate(x1, (32, 32), mode="bilinear", align_corners=False)
    x2 = F.interpolate(x2, (32, 32), mode="bilinear", align_corners=False)
    return ssim_matlab(x1, x2) < scdet_threshold


def to_tensor(img, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    return (
        torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device) / 255.0
    )


def to_cv2(img):
    return (img[0].cpu().float().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)


def get_valid_net_inp_size(img, scale, div=64):
    h, w, _ = img.shape
    src_h, src_w, _ = img.shape

    if h * scale % div != 0:
        h = (h * scale // div + 1) * div / scale
        h = int(h)

    if w * scale % div != 0:
        w = (w * scale // div + 1) * div / scale
        w = int(w)

    return {
        "src_size": (src_h, src_w),
        "dst_size": (h, w),
    }


def to_inp(npInp, dst_size):
    tenInp = to_tensor(npInp)
    tenInp = resize(tenInp, dst_size)
    return tenInp


def to_out(tenInp, src_size):
    tenInp = resize(tenInp, src_size)
    npOut = to_cv2(tenInp)
    return npOut


def resize(tensor, size):
    return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)


# Flow distance calculator


def distance_calculator(_x):
    dtype = _x.dtype
    u, v = _x[:, 0:1].float(), _x[:, 1:].float()
    return torch.sqrt(u**2 + v**2).to(dtype)


def convert(param):
    return {k.replace("module.", ""): v for k, v in param.items() if "module." in k}


def mark_tensor(tensor, text):
    """
    Mark something to tensor for debugging
    """
    n, c, h, w = tensor.shape
    to_pil = transforms.ToPILImage()
    image_pil = to_pil(tensor.clone().squeeze(0))

    draw = ImageDraw.Draw(image_pil)

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    text_width, text_height = draw.textsize(text, font=font)

    x_pos = w - text_width - 10
    y_pos = 10

    # 在图片上绘制文字
    draw.text((x_pos, y_pos), text, font=font, fill=(255, 255, 255))

    to_tensor = transforms.ToTensor()
    image_tensor_with_text = to_tensor(image_pil).unsqueeze(0)

    return image_tensor_with_text


class TMapper:
    def __init__(self, src=-1.0, dst=0.0, times=-1):
        self.times = dst / src if times == -1 else times
        self.now_step = -1

    def get_range_timestamps(
        self, _min: float, _max: float, lclose=True, rclose=False, normalize=True
    ) -> list:
        _min_step = math.ceil(_min * self.times)
        _max_step = math.ceil(_max * self.times)
        _start = _min_step if lclose else _min_step + 1
        _end = _max_step if not rclose else _max_step + 1
        if _start >= _end:
            return []
        if normalize:
            return [
                ((_i / self.times) - _min) / (_max - _min) for _i in range(_start, _end)
            ]
        return [_i / self.times for _i in range(_start, _end)]


ones_cache = {}


def get_ones_tensor(tensor: torch.Tensor):
    k = (str(tensor.device), str(tensor.size()))
    if k in ones_cache:
        return ones_cache[k]
    ones_cache[k] = torch.ones(
        tensor.size(), requires_grad=False, dtype=tensor.dtype
    ).to(tensor.device)
    return ones_cache[k]


def get_ones_tensor_size(size: tuple, device, dtype: torch.dtype):
    k = (str(device), str(size))
    if k in ones_cache:
        return ones_cache[k]
    ones_cache[k] = torch.ones(size, requires_grad=False, dtype=dtype).to(device)
    return ones_cache[k]


class VideoFI_IO:
    def __init__(self, input_path, output_path, dst_fps=60, times=-1, hwaccel=False):
        self.video_capture = cv2.VideoCapture(input_path)
        self.src_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.dst_fps = dst_fps
        if times != -1:
            self.dst_fps = times * self.src_fps
        self.total_frames_count = self.video_capture.get(7)
        self.width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.ffmpeg_writer = self.generate_frame_renderer(
            input_path, output_path, self.width, self.height, self.dst_fps, hwaccel
        )
        self.read_buffer = Queue(maxsize=100)
        self.write_buffer = Queue(maxsize=-1)
        _thread.start_new_thread(
            self.build_read_buffer, (self.read_buffer, self.video_capture)
        )
        _thread.start_new_thread(self.clear_write_buffer, (self.write_buffer,))

    def generate_frame_renderer(
        self, input_path, output_path, width, height, dst_fps, hwaccel=False
    ):
        encoder = "libx264"
        preset = "medium"
        if hwaccel:
            encoder = "h264_nvenc"
            preset = "p7"
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-r",
            f"{dst_fps}",
            "-s",
            f"{width}x{height}",
            "-i",
            "pipe:0",
            "-i",
            input_path,
            "-map",
            "0:v",
            "-map",
            "1:a?",
            "-c:v",
            encoder,
            "-movflags",
            "+faststart",
            "-pix_fmt",
            "yuv420p",
            "-qp",
            "16",
            "-preset",
            preset,
            "-c:a",
            "aac",
            "-b:a",
            "320k",
            f"{output_path}",
        ]

        return subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    def build_read_buffer(self, r_buffer, v):
        ret, __x = v.read()
        while ret:
            r_buffer.put(__x)
            ret, __x = v.read()
        r_buffer.put(None)

    def clear_write_buffer(self, w_buffer):
        while True:
            item = w_buffer.get()
            if item is None:
                break
            self.ffmpeg_writer.stdin.write(np.ascontiguousarray(item[:, :, ::-1]))
        self.ffmpeg_writer.stdin.close()
        self.ffmpeg_writer.wait()

    def write_frame(self, x):
        self.write_buffer.put(x)

    def read_frame(self):
        return self.read_buffer.get()

    def finish_writing(self):
        return self.write_buffer.empty() or self.ffmpeg_writer.stdin.closed
