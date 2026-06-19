import math
import pathlib
from collections.abc import Generator
from time import sleep

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from src.logic.handlers.backend.backend_handler import BackendHandler
from src.logic.render.backends.pytorch.InterpolateArchs.DetectInterpolateArch import (
    ArchDetect,
)
from src.logic.render.methods.interpolate.interpolate_method_base import (
    InterpolateMethodBase,
)
from src.schemas.domain import Frame, InterpolateModel
from src.schemas.domain.video_info import InputVideoInfo
from src.utils.LogConfig import get_logger
from src.utils.Util import errorAndLog

logger = get_logger(__name__)

torch.set_float32_matmul_precision("medium")
torch.set_grad_enabled(False)


class InterpolatePyTorch(InterpolateMethodBase):
    @torch.inference_mode()
    def __init__(
        self,
        backend_handler: BackendHandler,
        interpolate_model: InterpolateModel,
        input_video_info: InputVideoInfo,
    ):
        self.interpolate_model = interpolate_model
        self.ceilInterpolateFactor = interpolate_model.interpolate_factor
        self.width = input_video_info.width
        self.height = input_video_info.height
        self.hdr_mode = input_video_info.is_hdr
        self.bit_depth = input_video_info.bit_depth

        self.frame0 = None
        self.encode0 = None
        self._save_counter = 0
        self._output_dir = pathlib.Path("/tmp/debug_frames")
        self._output_dir.mkdir(parents=True, exist_ok=True)

        accelerator = interpolate_model.backend.accelerator
        if accelerator == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda", 0)
        elif accelerator == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        precision_id = interpolate_model.precision.precision_id
        self.dtype = getattr(torch, precision_id, torch.float32)

    def _bytes_to_tensor(self, data: bytes) -> torch.Tensor:
        src_dtype = torch.uint16 if self.hdr_mode else torch.uint8
        t = torch.frombuffer(data, dtype=src_dtype)
        t = (
            t.to(device=self.device)
            .div(65535.0 if self.hdr_mode else 255.0)
            .clamp(0.0, 1.0)
            .reshape(self.height, self.width, 3)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .contiguous()
            .to(dtype=self.dtype)
        )
        return t

    def _tensor_to_bytes(self, t: torch.Tensor) -> bytes:
        t = (
            t.squeeze(0)
            .permute(1, 2, 0)
            .clamp(0.0, 1.0)
            .mul(65535.0 if self.hdr_mode else 255.0)
            .round()
            .to(torch.uint16 if self.hdr_mode else torch.uint8)
            .contiguous()
            .detach()
            .cpu()
        )
        return t.numpy().tobytes()

    @torch.inference_mode()
    def _load(self):
        state_dict = torch.load(
            self.interpolate_model.file_path,
            map_location=self.device,
            weights_only=True,
            mmap=True,
        )

        ad = ArchDetect(self.interpolate_model.file_path)
        interpolateArch = ad.getArchName()
        _pad = 32
        self.encode = None
        scale = 1.0
        ensemble = False

        match interpolateArch.lower():
            case "rife46":
                from src.logic.render.backends.pytorch.InterpolateArchs.RIFE.rife46IFNET import (
                    IFNet,
                )
            case "rife47":
                from src.logic.render.backends.pytorch.InterpolateArchs.RIFE.rife47IFNET import (
                    IFNet,
                )

                self.encode = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 16, 3, 2, 1),
                    torch.nn.ConvTranspose2d(16, 4, 4, 2, 1),
                ).float()
            case "rife413":
                from src.logic.render.backends.pytorch.InterpolateArchs.RIFE.rife413IFNET import (
                    Head,
                    IFNet,
                )

                self.encode = Head()
            case "rife420":
                from src.logic.render.backends.pytorch.InterpolateArchs.RIFE.rife420IFNET import (
                    Head,
                    IFNet,
                )

                self.encode = Head()
            case "rife421":
                from src.logic.render.backends.pytorch.InterpolateArchs.RIFE.rife421IFNET import (
                    Head,
                    IFNet,
                )

                self.encode = Head()
            case "rife422lite":
                from src.logic.render.backends.pytorch.InterpolateArchs.RIFE.rife422_liteIFNET import (
                    Head,
                    IFNet,
                )

                self.encode = Head()
            case "rife425":
                from src.logic.render.backends.pytorch.InterpolateArchs.RIFE.rife425IFNET import (
                    Head,
                    IFNet,
                )

                _pad = 64
                self.encode = Head()
            case "rife425_heavy":
                from src.logic.render.backends.pytorch.InterpolateArchs.RIFE.rife425_heavyIFNET import (
                    Head,
                    IFNet,
                )

                _pad = 64
                self.encode = Head()
            case _:
                errorAndLog("Invalid Interpolation Arch")
                exit()

        tmp = max(_pad, int(_pad / scale))

        self.pw = math.ceil(self.width / tmp) * tmp
        self.ph = math.ceil(self.height / tmp) * tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)

        self.tenFlow_div = torch.tensor(
            [(self.pw - 1.0) / 2.0, (self.ph - 1.0) / 2.0],
            dtype=torch.float32,
            device=self.device,
        )
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, self.pw, dtype=torch.float32, device=self.device)
            .view(1, 1, 1, self.pw)
            .expand(-1, -1, self.ph, -1)
        ).to(dtype=torch.float32, device=self.device)
        tenVertical = (
            torch.linspace(-1.0, 1.0, self.ph, dtype=torch.float32, device=self.device)
            .view(1, 1, self.ph, 1)
            .expand(-1, -1, -1, self.pw)
        ).to(dtype=torch.float32, device=self.device)
        self.backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1)

        self.flownet = IFNet(scale=scale, ensemble=ensemble)

        state_dict = {
            k.replace("module.", ""): v for k, v in state_dict.items() if "module." in k
        }
        head_state_dict = {
            k.replace("encode.", ""): v for k, v in state_dict.items() if "encode." in k
        }
        if self.encode:
            self.encode.load_state_dict(state_dict=head_state_dict, strict=True)
            self.encode.eval().to(device=self.device, dtype=self.dtype)
        self.flownet.load_state_dict(state_dict=state_dict, strict=False)
        self.flownet.eval().to(device=self.device, dtype=self.dtype)

        self.timestepDict = {}
        for n in range(self.ceilInterpolateFactor):
            timestep = n / (self.ceilInterpolateFactor)
            timestep_tens = torch.full(
                (1, 1, self.ph, self.pw),
                timestep,
                dtype=self.dtype,
                device=self.device,
            )
            self.timestepDict[timestep] = timestep_tens

    def _debug_save_frame(self, frame):
        np_frame = frame.get_frame_np()
        if np_frame.dtype == np.uint16:
            np_frame = (
                np.clip(np_frame.astype(np.float32) / 65535.0, 0, 1) * 255
            ).astype(np.uint8)
        bgr = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
        self._save_counter += 1
        cv2.imwrite(
            str(self._output_dir / f"frame_{self._save_counter:06d}.jpg"),
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

    @torch.inference_mode()
    def process_frame(
        self,
        frame: Frame,
        transition=False,
    ) -> Generator[Frame, None, None]:
        frame_tensor = self._bytes_to_tensor(frame.get_frame_bytes())

        if self.frame0 is None:
            self.frame0 = F.pad(frame_tensor, self.padding)
            if self.encode:
                self.encode0 = self.encode(self.frame0)
            return

        frame1 = F.pad(frame_tensor, self.padding)

        if self.encode:
            encode1 = self.encode(frame1)

        for n in range(self.ceilInterpolateFactor - 1):
            if not transition:
                timestep = (n + 1) * 1.0 / (self.ceilInterpolateFactor)
                while self.flownet is None:
                    sleep(1)
                timestep = self.timestepDict[timestep]

                if self.encode:
                    output = self.flownet(
                        self.frame0,
                        frame1,
                        timestep,
                        self.tenFlow_div,
                        self.backwarp_tenGrid,
                        self.encode0,
                        encode1,
                    )
                else:
                    output = self.flownet(
                        self.frame0,
                        frame1,
                        timestep,
                        self.tenFlow_div,
                        self.backwarp_tenGrid,
                    )

                crop = output[:, :, : self.height, : self.width]
                out_bytes = self._tensor_to_bytes(crop)
                ret_frame = frame.get_dummy_frame()
                ret_frame.set_frame_bytes(out_bytes)
                yield ret_frame
            else:
                yield frame

        self.frame0 = frame1
        if self.encode:
            self.encode0 = encode1
