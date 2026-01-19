# for study only
from models.model_gmfss.GMFSS import Model
from models.drm import calc_drm_gmfss
import torch


class GMFSS:
    def __init__(
        self,
        weights=r"weights/train_log_gmfss",
        scale=1.0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.model = Model()
        self.model.load_model(weights, -1)
        self.model.device(device)
        self.model.eval()
        self.scale = scale
        self.pad_size = 64

    @torch.inference_mode()
    @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    def inference_ts(self, I0, I1, ts):
        reuse = self.model.reuse(I0, I1, self.scale)

        output = []
        for t in ts:
            if t == 0:
                output.append(I0)
            elif t == 1:
                output.append(I1)
            else:
                output.append(
                    self.model.inference(I0, I1, reuse, timestep0=t, timestep1=1 - t)
                )

        return output

    @torch.inference_mode()
    @torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
    def inference_ts_drba(self, I0, I1, I2, ts, reuse=None, linear=False):
        reuseI1I0 = self.model.reuse(I1, I0, self.scale) if reuse is None else reuse
        reuseI1I2 = self.model.reuse(I1, I2, self.scale)

        flow10, metric10 = reuseI1I0[0], reuseI1I0[2]
        flow12, metric12 = reuseI1I2[0], reuseI1I2[2]

        output = []

        for t in ts:
            if t == 0:
                output.append(I0)
            elif t == 1:
                output.append(I1)
            elif t == 2:
                output.append(I2)
            elif 0 < t < 1:
                t = 1 - t
                drm = calc_drm_gmfss(t, flow10, flow12, metric10, metric12, linear)
                out = self.model.inference(
                    I1,
                    I0,
                    reuseI1I0,
                    timestep0=drm["drm1t_t01"],
                    timestep1=drm["drm0t_t01"],
                )
                output.append(out)

            elif 1 < t < 2:
                t = t - 1
                drm = calc_drm_gmfss(t, flow10, flow12, metric10, metric12, linear)
                out = self.model.inference(
                    I1,
                    I2,
                    reuseI1I2,
                    timestep0=drm["drm1t_t12"],
                    timestep1=drm["drm2t_t12"],
                )
                output.append(out)

        # next reuseI1i0 = reverse(current reuseI1i2)
        # f0, f1, m0, m1, feat0, feat1 = reuseI1i2
        # _reuse = (f1, f0, m1, m0, feat1, feat0)
        reuse = [
            value for pair in zip(reuseI1I2[1::2], reuseI1I2[0::2]) for value in pair
        ]

        return output, reuse
