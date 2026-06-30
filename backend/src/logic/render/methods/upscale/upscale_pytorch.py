import math
from collections.abc import Generator
from time import sleep

import torch
import torch.nn.functional as F

from src.logic.render.backends.pytorch.TorchUtils import TorchUtils
from src.logic.render.backends.pytorch.UpscaleModelWrapper import UpscaleModelWrapper
from src.schemas.domain import Frame, UpscaleModel
from src.schemas.domain.video_info import InputVideoInfo
from src.utils.LogConfig import get_logger

logger = get_logger(__name__)

torch.set_float32_matmul_precision("medium")
torch.set_grad_enabled(False)


class UpscalePyTorch:
    @torch.inference_mode()
    def __init__(
        self,
        backend_handler: TorchUtils,
        upscale_model: UpscaleModel,
        input_video_info: InputVideoInfo,
    ):
        self.upscale_model = upscale_model
        self.width = input_video_info.width
        self.height = input_video_info.height
        self.hdr_mode = input_video_info.is_hdr

        self.tile_pad = 10
        self.tile = [0, 0]

    @torch.inference_mode()
    def _load(self):
        self.upscale_model_wrapper = UpscaleModelWrapper(
            model_path=self.upscale_model.file_path,
            device=self.device,
            precision=self.dtype,
        )
        self.scale = self.upscale_model_wrapper.get_scale()

        match self.scale:
            case 1:
                modulo = 4
            case 2:
                modulo = 2
            case _:
                modulo = 1
        if all(t > 0 for t in self.tile):
            self.pad_w = (
                math.ceil(min(self.tile[0] + 2 * self.tile_pad, self.width) / modulo)
                * modulo
            )
            self.pad_h = (
                math.ceil(min(self.tile[1] + 2 * self.tile_pad, self.height) / modulo)
                * modulo
            )
        else:
            modulo = 1 if self.width < 720 or self.height < 720 else 1
            self.pad_w = math.ceil(self.width / modulo) * modulo
            self.pad_h = math.ceil(self.height / modulo) * modulo

    def _render_tiled_image(self, img: torch.Tensor) -> torch.Tensor:
        scale = self.scale
        tile = self.tile
        tile_pad = self.tile_pad

        batch, channel, height, width = img.shape
        output_shape = (batch, channel, height * scale, width * scale)

        output = img.new_zeros(output_shape).to(device=self.device, dtype=self.dtype)

        tiles_x = math.ceil(width / tile[0])
        tiles_y = math.ceil(height / tile[1])

        for y in range(tiles_y):
            for x in range(tiles_x):
                ofs_x = x * tile[0]
                ofs_y = y * tile[1]

                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile[0], width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile[1], height)

                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y

                input_tile = img[
                    :,
                    :,
                    input_start_y_pad:input_end_y_pad,
                    input_start_x_pad:input_end_x_pad,
                ].to(device=self.device, dtype=self.dtype)

                h, w = input_tile.shape[2:]
                input_tile = F.pad(
                    input_tile,
                    (0, self.pad_w - w, 0, self.pad_h - h),
                    "replicate",
                )

                output_tile = self.upscale_model_wrapper(input_tile)

                output_tile = output_tile[:, :, : h * scale, : w * scale]

                output_start_x = input_start_x * scale
                output_end_x = input_end_x * scale
                output_start_y = input_start_y * scale
                output_end_y = input_end_y * scale

                output_start_x_tile = (input_start_x - input_start_x_pad) * scale
                output_end_x_tile = output_start_x_tile + input_tile_width * scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * scale
                output_end_y_tile = output_start_y_tile + input_tile_height * scale

                output[
                    :,
                    :,
                    output_start_y:output_end_y,
                    output_start_x:output_end_x,
                ] = output_tile[
                    :,
                    :,
                    output_start_y_tile:output_end_y_tile,
                    output_start_x_tile:output_end_x_tile,
                ]

        return output

    @torch.inference_mode()
    def process_frame(
        self,
        frame: Frame,
    ) -> Generator[Frame, None, None]:
        frame_tensor = frame.bytes_to_tensor(self.device, self.dtype)

        while self.upscale_model_wrapper is None:
            sleep(1)

        if self.tile[0] == 0 and self.tile[1] == 0:
            output = self.upscale_model_wrapper(frame_tensor)
        else:
            output = self._render_tiled_image(frame_tensor)

        scale = self.scale
        crop = output[:, :, : self.height * scale, : self.width * scale]
        out_width = self.width * scale
        out_height = self.height * scale
        out_bytes = frame.tensor_to_bytes(crop)
        ret_frame = frame.get_dummy_frame()
        ret_frame.width = out_width
        ret_frame.height = out_height
        ret_frame.set_frame_bytes(out_bytes)
        yield ret_frame
