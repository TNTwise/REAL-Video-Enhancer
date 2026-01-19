import torch
import torch.nn.functional as F
from models.utils.tools import check_cupy_env
from models.gmflow.gmflow import GMFlow
from models.model_gmfss_union.MetricNet import MetricNet
from models.model_gmfss_union.FeatureNet import FeatureNet
from models.model_gmfss_union.FusionNet import GridNet

if check_cupy_env():
    from models.softsplat.softsplat import softsplat as warp
else:
    print("System does not have CUDA installed, falling back to PyTorch")
    from models.softsplat.softsplat_torch import softsplat as warp


class Model:
    def __init__(self):
        self.flownet = GMFlow()
        self.metricnet = MetricNet()
        self.feat_ext = FeatureNet()
        self.fusionnet = GridNet(9, 64 * 2, 128 * 2, 192 * 2, 3)
        self.version = 3.9

    def half(self):
        self.flownet.half()
        self.metricnet.half()
        self.feat_ext.half()
        self.fusionnet.half()

    def eval(self):
        self.flownet.eval()
        self.metricnet.eval()
        self.feat_ext.eval()
        self.fusionnet.eval()

    def device(
        self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.flownet.to(device)
        self.metricnet.to(device)
        self.feat_ext.to(device)
        self.fusionnet.to(device)

    def load_model(self, path, rank):
        def convert(param):
            return {
                k.replace("module.", ""): v for k, v in param.items() if "module." in k
            }

        self.flownet.load_state_dict(torch.load("{}/flownet.pkl".format(path)))
        self.metricnet.load_state_dict(torch.load("{}/metric.pkl".format(path)))
        self.feat_ext.load_state_dict(torch.load("{}/feat.pkl".format(path)))
        self.fusionnet.load_state_dict(torch.load("{}/fusionnet.pkl".format(path)))

    def reuse(self, img0, img1, scale):
        feat11, feat12, feat13 = self.feat_ext(img0)
        feat21, feat22, feat23 = self.feat_ext(img1)
        feat_ext0 = [feat11, feat12, feat13]
        feat_ext1 = [feat21, feat22, feat23]

        img0 = F.interpolate(
            img0, scale_factor=0.5, mode="bilinear", align_corners=False
        )
        img1 = F.interpolate(
            img1, scale_factor=0.5, mode="bilinear", align_corners=False
        )

        if scale != 1.0:
            imgf0 = F.interpolate(
                img0, scale_factor=scale, mode="bilinear", align_corners=False
            )
            imgf1 = F.interpolate(
                img1, scale_factor=scale, mode="bilinear", align_corners=False
            )
        else:
            imgf0 = img0
            imgf1 = img1
        flow01 = self.flownet(imgf0, imgf1)
        flow10 = self.flownet(imgf1, imgf0)
        if scale != 1.0:
            flow01 = (
                F.interpolate(
                    flow01,
                    scale_factor=1.0 / scale,
                    mode="bilinear",
                    align_corners=False,
                )
                / scale
            )
            flow10 = (
                F.interpolate(
                    flow10,
                    scale_factor=1.0 / scale,
                    mode="bilinear",
                    align_corners=False,
                )
                / scale
            )

        metric0, metric1 = self.metricnet(img0, img1, flow01, flow10)

        return flow01, flow10, metric0, metric1, feat_ext0, feat_ext1

    def inference(
        self, img0, img1, reuse_things, timestep0, timestep1, rife, enable_mask=True
    ):
        flow01, metric0, feat11, feat12, feat13 = (
            reuse_things[0],
            reuse_things[2],
            reuse_things[4][0],
            reuse_things[4][1],
            reuse_things[4][2],
        )
        flow10, metric1, feat21, feat22, feat23 = (
            reuse_things[1],
            reuse_things[3],
            reuse_things[5][0],
            reuse_things[5][1],
            reuse_things[5][2],
        )

        F1t = timestep0 * flow01
        F2t = timestep1 * flow10

        Z1t = timestep0 * metric0
        Z2t = timestep1 * metric1

        img0 = F.interpolate(
            img0, scale_factor=0.5, mode="bilinear", align_corners=False
        )
        I1t = warp(img0, F1t, Z1t, strMode="soft")
        img1 = F.interpolate(
            img1, scale_factor=0.5, mode="bilinear", align_corners=False
        )
        I2t = warp(img1, F2t, Z2t, strMode="soft")

        feat1t1 = warp(feat11, F1t, Z1t, strMode="soft")
        feat2t1 = warp(feat21, F2t, Z2t, strMode="soft")

        F1td = (
            F.interpolate(F1t, scale_factor=0.5, mode="bilinear", align_corners=False)
            * 0.5
        )
        Z1d = F.interpolate(Z1t, scale_factor=0.5, mode="bilinear", align_corners=False)
        feat1t2 = warp(feat12, F1td, Z1d, strMode="soft")
        F2td = (
            F.interpolate(F2t, scale_factor=0.5, mode="bilinear", align_corners=False)
            * 0.5
        )
        Z2d = F.interpolate(Z2t, scale_factor=0.5, mode="bilinear", align_corners=False)
        feat2t2 = warp(feat22, F2td, Z2d, strMode="soft")

        F1tdd = (
            F.interpolate(F1t, scale_factor=0.25, mode="bilinear", align_corners=False)
            * 0.25
        )
        Z1dd = F.interpolate(
            Z1t, scale_factor=0.25, mode="bilinear", align_corners=False
        )
        feat1t3 = warp(feat13, F1tdd, Z1dd, strMode="soft")
        F2tdd = (
            F.interpolate(F2t, scale_factor=0.25, mode="bilinear", align_corners=False)
            * 0.25
        )
        Z2dd = F.interpolate(
            Z2t, scale_factor=0.25, mode="bilinear", align_corners=False
        )
        feat2t3 = warp(feat23, F2tdd, Z2dd, strMode="soft")

        if isinstance(timestep0, torch.Tensor) and enable_mask:
            # Warp the input timestep to align with the warped image
            timestep0 = warp(timestep0, F1t, Z1t, strMode="soft")
            timestep1 = warp(timestep1, F2t, Z2t, strMode="soft")

            # Fill in the holes
            gaps0 = warp(timestep0.clone() * 0 + 1, F1t, Z1t, strMode="soft") < 0.999
            gaps1 = warp(timestep1.clone() * 0 + 1, F2t, Z2t, strMode="soft") < 0.999
            invalid_mask = torch.logical_or(gaps0, gaps1)
            timestep0[invalid_mask] = 1
            timestep1[invalid_mask] = 1

            # Create masks for swapping
            def get_mask(c, _scale=1.0):
                timestep0f, timestep1f = timestep0.clone(), timestep1.clone()
                if _scale != 1.0:
                    timestep0f = F.interpolate(
                        timestep0f,
                        scale_factor=_scale,
                        mode="bilinear",
                        align_corners=False,
                    )
                    timestep1f = F.interpolate(
                        timestep1f,
                        scale_factor=_scale,
                        mode="bilinear",
                        align_corners=False,
                    )
                # 25 is a hyperparameter, it was determined through experimentation.
                # Using this mask helps reduce the artifacts when encountering scene changes.
                _mask0 = torch.Tensor(timestep0f / timestep1f > 25).repeat(1, c, 1, 1)
                _mask1 = torch.Tensor(timestep1f / timestep0f > 25).repeat(1, c, 1, 1)
                return _mask0, _mask1

            # Swap regions with smaller timestep in the warped images/features (Smaller timestep correspond
            # to fewer artifacts when encountering scene changes)
            mask0, mask1 = get_mask(3, 1)
            I1t[mask0], I2t[mask1] = I2t[mask0], I1t[mask1]

            mask0, mask1 = get_mask(64, 1)
            feat1t1[mask0], feat2t1[mask1] = feat2t1[mask0], feat1t1[mask1]

            mask0, mask1 = get_mask(128, 0.5)
            feat1t2[mask0], feat2t2[mask1] = feat2t2[mask0], feat1t2[mask1]

            mask0, mask1 = get_mask(192, 0.25)
            feat1t3[mask0], feat2t3[mask1] = feat2t3[mask0], feat1t3[mask1]

        out = self.fusionnet(
            torch.cat([I1t, rife, I2t], dim=1),
            torch.cat([feat1t1, feat2t1], dim=1),
            torch.cat([feat1t2, feat2t2], dim=1),
            torch.cat([feat1t3, feat2t3], dim=1),
        )

        return torch.clamp(out, 0, 1)

    # def inference(self, img0, img1, reuse_things, timestep, rife):
    #     flow01, metric0, feat11, feat12, feat13 = reuse_things[0], reuse_things[2], reuse_things[4][0], reuse_things[4][
    #         1], reuse_things[4][2]
    #     flow10, metric1, feat21, feat22, feat23 = reuse_things[1], reuse_things[3], reuse_things[5][0], reuse_things[5][
    #         1], reuse_things[5][2]
    #
    #     F1t = timestep * flow01
    #     F2t = (1 - timestep) * flow10
    #
    #     Z1t = timestep * metric0
    #     Z2t = (1 - timestep) * metric1
    #
    #     img0 = F.interpolate(img0, scale_factor=0.5, mode="bilinear", align_corners=False)
    #     I1t = warp(img0, F1t, Z1t, strMode='soft')
    #     img1 = F.interpolate(img1, scale_factor=0.5, mode="bilinear", align_corners=False)
    #     I2t = warp(img1, F2t, Z2t, strMode='soft')
    #
    #     feat1t1 = warp(feat11, F1t, Z1t, strMode='soft')
    #     feat2t1 = warp(feat21, F2t, Z2t, strMode='soft')
    #
    #     F1td = F.interpolate(F1t, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
    #     Z1d = F.interpolate(Z1t, scale_factor=0.5, mode="bilinear", align_corners=False)
    #     feat1t2 = warp(feat12, F1td, Z1d, strMode='soft')
    #     F2td = F.interpolate(F2t, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
    #     Z2d = F.interpolate(Z2t, scale_factor=0.5, mode="bilinear", align_corners=False)
    #     feat2t2 = warp(feat22, F2td, Z2d, strMode='soft')
    #
    #     F1tdd = F.interpolate(F1t, scale_factor=0.25, mode="bilinear", align_corners=False) * 0.25
    #     Z1dd = F.interpolate(Z1t, scale_factor=0.25, mode="bilinear", align_corners=False)
    #     feat1t3 = warp(feat13, F1tdd, Z1dd, strMode='soft')
    #     F2tdd = F.interpolate(F2t, scale_factor=0.25, mode="bilinear", align_corners=False) * 0.25
    #     Z2dd = F.interpolate(Z2t, scale_factor=0.25, mode="bilinear", align_corners=False)
    #     feat2t3 = warp(feat23, F2tdd, Z2dd, strMode='soft')
    #
    #     out = self.fusionnet(torch.cat([I1t, rife, I2t], dim=1), torch.cat([feat1t1, feat2t1], dim=1),
    #                          torch.cat([feat1t2, feat2t2], dim=1), torch.cat([feat1t3, feat2t3], dim=1))
    #
    #     return torch.clamp(out, 0, 1)
