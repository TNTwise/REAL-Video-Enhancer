import torch
import torch.nn.functional as F
from models.gmflow.gmflow import GMFlow
from models.model_gmfss.FeatureNet import FeatureNet
from models.model_gmfss.FusionNet import GridNet
from models.model_gmfss.MetricNet import MetricNet
from models.utils.tools import check_cupy_env

if check_cupy_env():
    from models.softsplat.softsplat import softsplat as warp
else:
    print('System does not have CUDA installed, falling back to PyTorch')
    from models.softsplat.softsplat_torch import softsplat as warp


class Model:
    def __init__(self):
        self.flownet = GMFlow()
        self.metricnet = MetricNet()
        self.feat_ext = FeatureNet()
        self.fusionnet = GridNet(6 * 2, 64 * 2, 128 * 2, 192 * 2, 3)
        self.version = 3.9

    def eval(self):
        self.flownet.eval()
        self.metricnet.eval()
        self.feat_ext.eval()
        self.fusionnet.eval()

    def device(
        self,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        self.flownet.to(device)
        self.metricnet.to(device)
        self.feat_ext.to(device)
        self.fusionnet.to(device)

    def half(self):
        self.flownet.half()
        self.metricnet.half()
        self.feat_ext.half()
        self.fusionnet.half()

    def load_model(self, path, rank):
        def convert(param):
            return {
                k.replace('module.', ''): v
                for k, v in param.items()
                if 'module.' in k
            }

        self.flownet.load_state_dict(
            torch.load(f'{path}/flownet.pkl', map_location='cpu')
        )
        self.metricnet.load_state_dict(
            torch.load(f'{path}/metric.pkl', map_location='cpu')
        )
        self.feat_ext.load_state_dict(
            torch.load(f'{path}/feat.pkl', map_location='cpu')
        )
        self.fusionnet.load_state_dict(
            torch.load(f'{path}/fusionnet.pkl', map_location='cpu')
        )

    def reuse(self, img0, img1, scale):
        feat11, feat12, feat13 = self.feat_ext(img0)
        feat21, feat22, feat23 = self.feat_ext(img1)
        feat_ext0 = [feat11, feat12, feat13]
        feat_ext1 = [feat21, feat22, feat23]

        img0 = F.interpolate(
            img0, scale_factor=0.5, mode='bilinear', align_corners=False
        )
        img1 = F.interpolate(
            img1, scale_factor=0.5, mode='bilinear', align_corners=False
        )

        if scale != 1.0:
            imgf0 = F.interpolate(
                img0, scale_factor=scale, mode='bilinear', align_corners=False
            )
            imgf1 = F.interpolate(
                img1, scale_factor=scale, mode='bilinear', align_corners=False
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
                    mode='bilinear',
                    align_corners=False,
                )
                / scale
            )
            flow10 = (
                F.interpolate(
                    flow10,
                    scale_factor=1.0 / scale,
                    mode='bilinear',
                    align_corners=False,
                )
                / scale
            )

        metric0, metric1 = self.metricnet(img0, img1, flow01, flow10)

        return flow01, flow10, metric0, metric1, feat_ext0, feat_ext1

    def inference(
        self, img0, img1, reuse_things, timestep0, timestep1, swap_thresh=1
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
            img0, scale_factor=0.5, mode='bilinear', align_corners=False
        )
        I1t = warp(img0, F1t, Z1t, strMode='soft')
        img1 = F.interpolate(
            img1, scale_factor=0.5, mode='bilinear', align_corners=False
        )
        I2t = warp(img1, F2t, Z2t, strMode='soft')

        feat1t1 = warp(feat11, F1t, Z1t, strMode='soft')
        feat2t1 = warp(feat21, F2t, Z2t, strMode='soft')

        F1td = (
            F.interpolate(
                F1t, scale_factor=0.5, mode='bilinear', align_corners=False
            )
            * 0.5
        )
        Z1d = F.interpolate(
            Z1t, scale_factor=0.5, mode='bilinear', align_corners=False
        )
        feat1t2 = warp(feat12, F1td, Z1d, strMode='soft')
        F2td = (
            F.interpolate(
                F2t, scale_factor=0.5, mode='bilinear', align_corners=False
            )
            * 0.5
        )
        Z2d = F.interpolate(
            Z2t, scale_factor=0.5, mode='bilinear', align_corners=False
        )
        feat2t2 = warp(feat22, F2td, Z2d, strMode='soft')

        F1tdd = (
            F.interpolate(
                F1t, scale_factor=0.25, mode='bilinear', align_corners=False
            )
            * 0.25
        )
        Z1dd = F.interpolate(
            Z1t, scale_factor=0.25, mode='bilinear', align_corners=False
        )
        feat1t3 = warp(feat13, F1tdd, Z1dd, strMode='soft')
        F2tdd = (
            F.interpolate(
                F2t, scale_factor=0.25, mode='bilinear', align_corners=False
            )
            * 0.25
        )
        Z2dd = F.interpolate(
            Z2t, scale_factor=0.25, mode='bilinear', align_corners=False
        )
        feat2t3 = warp(feat23, F2tdd, Z2dd, strMode='soft')

        # swap_thresh means Threshold for applying the swap mask.
        # 0 means fully apply the swap mask.
        # 0.n means enable swapping when the timestep difference is greater than 0.n.
        # 1 means never apply the swap mask.
        # swap_thresh = 1

        # deprecated
        # if swap_thresh != 1:
        #     # Warp the input timesteps to align with the warped image
        #     timestep0 = warp(timestep0, F1t, Z1t, strMode='soft')
        #     timestep1 = warp(timestep1, F2t, Z2t, strMode='soft')
        #
        #     # Fill in the holes
        #     ones_mask0 = warp(torch.ones_like(timestep0).to(timestep0.device), F1t, Z1t, strMode='soft') < 0.999
        #     ones_mask1 = warp(torch.ones_like(timestep1).to(timestep1.device), F2t, Z2t, strMode='soft') < 0.999
        #     timestep0[ones_mask0] = 1
        #     timestep1[ones_mask1] = 1
        #
        #     # Create masks for swapping based on timestep differences and threshold
        #     def get_mask(c, _scale=1.0):
        #         timestep0f, timestep1f = timestep0.clone(), timestep1.clone()
        #         if _scale != 1.0:
        #             timestep0f = F.interpolate(timestep0f, scale_factor=_scale, mode="bilinear", align_corners=False)
        #             timestep1f = F.interpolate(timestep1f, scale_factor=_scale, mode="bilinear", align_corners=False)
        #         _mask0 = torch.Tensor(timestep0f.repeat(1, c, 1, 1) > timestep1f.repeat(1, c, 1, 1))
        #         _mask1 = torch.Tensor(timestep0f.repeat(1, c, 1, 1) < timestep1f.repeat(1, c, 1, 1))
        #         diff_thresh_mask = torch.Tensor(torch.abs(timestep0f - timestep1f) >= swap_thresh)
        #         _mask0 = torch.logical_and(_mask0, diff_thresh_mask)
        #         _mask1 = torch.logical_and(_mask1, diff_thresh_mask)
        #         return _mask0, _mask1
        #
        #     # Swap regions with smaller timestep in the warped images/features (Smaller timestep correspond
        #     # to smaller distortions and less disappearance, which improves accuracy for this task)
        #     mask0, mask1 = get_mask(3, 1)
        #     I1t[mask0], I2t[mask1] = I2t[mask0], I1t[mask1]
        #
        #     mask0, mask1 = get_mask(64, 1)
        #     feat1t1[mask0], feat2t1[mask1] = feat2t1[mask0], feat1t1[mask1]
        #
        #     mask0, mask1 = get_mask(128, 0.5)
        #     feat1t2[mask0], feat2t2[mask1] = feat2t2[mask0], feat1t2[mask1]
        #
        #     mask0, mask1 = get_mask(192, 0.25)
        #     feat1t3[mask0], feat2t3[mask1] = feat2t3[mask0], feat1t3[mask1]

        out = self.fusionnet(
            torch.cat([img0, I1t, I2t, img1], dim=1),
            torch.cat([feat1t1, feat2t1], dim=1),
            torch.cat([feat1t2, feat2t2], dim=1),
            torch.cat([feat1t3, feat2t3], dim=1),
        )

        # visualization experiment
        # import cv2
        # cv2.imwrite(r"E:\Work\VFI\Algorithm\FCLAFI\output\I1t.png", I1t[0].permute(1, 2, 0).cpu().numpy() * 255)
        # cv2.imwrite(r"E:\Work\VFI\Algorithm\FCLAFI\output\I2t.png", I2t[0].permute(1, 2, 0).cpu().numpy() * 255)
        #
        # wtimestep0 = warp(timestep0, F1t, Z1t, strMode='soft')
        # wtimestep1 = warp(timestep1, F2t, Z2t, strMode='soft')
        #
        # ones_mask0 = warp(torch.ones_like(timestep0).to(timestep0.device), F1t, Z1t, strMode='soft') < 0.999
        # ones_mask1 = warp(torch.ones_like(timestep1).to(timestep1.device), F2t, Z2t, strMode='soft') < 0.999
        #
        # cv2.imwrite(r"E:\Work\VFI\Algorithm\FCLAFI\output\omask0.png", ones_mask0[0].permute(1, 2, 0).cpu().numpy() * 255)
        # cv2.imwrite(r"E:\Work\VFI\Algorithm\FCLAFI\output\omask1.png", ones_mask1[0].permute(1, 2, 0).cpu().numpy() * 255)
        #
        # cv2.imwrite(r"E:\Work\VFI\Algorithm\FCLAFI\output\wot0.png", wtimestep0[0].permute(1, 2, 0).cpu().numpy() * 255)
        # cv2.imwrite(r"E:\Work\VFI\Algorithm\FCLAFI\output\wot1.png", wtimestep1[0].permute(1, 2, 0).cpu().numpy() * 255)
        #
        # wtimestep0[ones_mask0] = 1
        # wtimestep1[ones_mask1] = 1
        #
        # cv2.imwrite(r"E:\Work\VFI\Algorithm\FCLAFI\output\t0.png", timestep0[0].permute(1, 2, 0).cpu().numpy() * 255)
        # cv2.imwrite(r"E:\Work\VFI\Algorithm\FCLAFI\output\t1.png", timestep1[0].permute(1, 2, 0).cpu().numpy() * 255)
        # cv2.imwrite(r"E:\Work\VFI\Algorithm\FCLAFI\output\wt0.png", wtimestep0[0].permute(1, 2, 0).cpu().numpy() * 255)
        # cv2.imwrite(r"E:\Work\VFI\Algorithm\FCLAFI\output\wt1.png", wtimestep1[0].permute(1, 2, 0).cpu().numpy() * 255)

        return torch.clamp(out, 0, 1)

    # def inference(self, img0, img1, reuse_things, timestep):
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
    #     out = self.fusionnet(torch.cat([img0, I1t, I2t, img1], dim=1), torch.cat([feat1t1, feat2t1], dim=1),
    #                          torch.cat([feat1t2, feat2t2], dim=1), torch.cat([feat1t3, feat2t3], dim=1))
    #
    #     return torch.clamp(out, 0, 1)
