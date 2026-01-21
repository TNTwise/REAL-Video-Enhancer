import warnings

import numpy as np
import torch

from .models.utils.tools import (
    TMapper,
    check_scene,
    get_valid_net_inp_size,
    to_inp,
)

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


class DRBA_RVE:
    def __init__(self, model_type, model_path, dst_fps, times, scale, fps):
        self.model_type = model_type  # model network type
        self.scale = scale  # flow scale
        self.dst_fps = dst_fps
        self.times = times
        self.src_fps = fps
        self.model_path = model_path
        self.pad_size = 64
        self.enable_scdet = True
        self.scdet_threshold = 0.3
        self.idx = 0
        self.i0 = None
        self.i1 = None
        self.I0 = None
        self.I1 = None
        self.left_scene = False
        self.right_scene = False

        self.model = self.load_model(model_type)

    def load_model(self, model_type):
        if model_type == 'rife':
            from .models.rife import RIFE

            model = RIFE(
                weights=self.model_path, scale=self.scale, device=device
            )
        elif model_type == 'gmfss':
            from .models.gmfss import GMFSS

            model = GMFSS(
                weights=self.model_path, scale=self.scale, device=device
            )
        elif model_type == 'gmfss_union':
            from .models.gmfss_union import GMFSS_UNION

            model = GMFSS_UNION(
                weights=self.model_path, scale=self.scale, device=device
            )
        else:
            raise ValueError(f'model_type must in {model_type}')

        return model

    @classmethod
    def calc_t(cls, _idx: float):
        if cls.times != -1:
            if cls.times % 2:
                vfi_timestamp = [
                    (_i + 1) / cls.times for _i in range((cls.times - 1) // 2)
                ]  # 0 ~ 0.5
                vfi_timestamp = (
                    list(reversed([1 - t for t in vfi_timestamp]))
                    + [1]
                    + [t + 1 for t in vfi_timestamp]
                )
                return np.array(vfi_timestamp)
            vfi_timestamp = [
                (_i + 0.5) / cls.times for _i in range(cls.times // 2)
            ]  # 0 ~ 0.5
            vfi_timestamp = list(
                reversed([1 - t for t in vfi_timestamp])
            ) + [t + 1 for t in vfi_timestamp]
            return np.array(vfi_timestamp)

        timestamp = np.array(
            cls.t_mapper.get_range_timestamps(
                _idx - 0.5,
                _idx + 0.5,
                lclose=True,
                rclose=False,
                normalize=False,
            )
        )
        vfi_timestamp = np.round(timestamp - _idx, 4) + 1  # [0.5, 1.5)

        return vfi_timestamp

    def header(
        self,
        i0,
        i1,
    ):
        # start inference
        size = get_valid_net_inp_size(i0, self.scale, div=self.pad_size)
        self.src_size, self.dst_size = size['src_size'], size['dst_size']

        I0 = to_inp(i0, self.dst_size)
        I1 = to_inp(i1, self.dst_size)

        self.t_mapper = TMapper(self.src_fps, self.dst_fps, self.times)
        idx = 0

        # head
        ts = self.calc_t(idx)
        left_scene = (
            check_scene(I0, I1, self.scdet_threshold)
            if self.enable_scdet
            else False
        )
        right_scene = left_scene
        self.reuse = None

        if right_scene:
            output = [I0 for _ in ts]
        else:
            left_ts = ts[ts < 1]
            right_ts = ts[ts >= 1] - 1

            output = [I0 for _ in left_ts]
            output.extend(self.model.inference_ts(I0, I1, right_ts))

        for x in output:
            yield x

    def tail(self):
        # tail
        ts = self.calc_t(self.idx)
        left_ts = ts[ts <= 1]
        right_ts = ts[ts > 1] - 1

        output = self.model.inference_ts(self.I0, self.I1, left_ts)
        output.extend([self.I1 for _ in right_ts])

        for x in output:
            yield x
        self.idx += 1

    def inference(self, i2=None):
        if i2 is None:
            return
        I2 = to_inp(i2, self.dst_size)

        ts = self.calc_t(self.idx)
        self.right_scene = (
            check_scene(self.I1, I2, self.scdet_threshold)
            if self.enable_scdet
            else False
        )

        # If a scene transition occurs between the three frames, then the calculation of this DRM is meaningless.
        if (
            self.left_scene and self.right_scene
        ):  # scene transition occurs at I0~I1, also occurs at I1~I2
            output = [self.I1 for _ in ts]
            self.reuse = None

        elif (
            self.left_scene and not self.right_scene
        ):  # scene transition occurs at I0~I1
            left_ts = ts[ts < 1]
            right_ts = ts[ts >= 1] - 1
            self.reuse = None

            output = [self.I1 for _ in left_ts]
            output.extend(self.model.inference_ts(self.I1, I2, right_ts))

        elif (
            not self.left_scene and self.right_scene
        ):  # scene transition occurs at I1~I2
            left_ts = ts[ts <= 1]
            right_ts = ts[ts > 1] - 1
            self.reuse = None

            output = self.model.inference_ts(self.I0, self.I1, left_ts)
            output.extend([self.I1 for _ in right_ts])

        else:  # no scene transition
            output, self.reuse = self.model.inference_ts_drba(
                self.I0, self.I1, I2, ts, self.reuse, linear=True
            )

        # debug
        # for i in range(len(output)):
        #     output[i] = mark_tensor(output[i], f"{ts[i] + idx}")

        for x in output:
            yield x

        self.i0, self.i1 = self.i1, i2
        self.I0, self.I1 = self.I1, I2
        self.left_scene = self.right_scene
        self.idx += 1
