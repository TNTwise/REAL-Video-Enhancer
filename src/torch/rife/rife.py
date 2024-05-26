import os
import torch
import numpy as np

try:
    import src.programData.thisdir

    thisdir = src.programData.thisdir.thisdir()
except:
    thisdir = f"{os.path.expanduser(r'~')}/.local/share/REAL-Video-Enhancer"
from torch.nn import functional as F


class Rife:
    def __init__(
        self,
        interpolation_factor,
        half,
        width,
        height,
        interpolate_method,
        ensemble=False,
        nt=1,
        UHD=False,
    ):
        self.interpolation_factor = interpolation_factor

        self.half = half

        self.scale = 1.0
        self.width = width
        self.height = height
        self.interpolate_method = interpolate_method
        self.ensemble = ensemble
        self.nt = nt

        self.UHD = self.width > 1920 or self.height > 1080

        self.handle_model()

    def handle_model(self):
        if self.interpolate_method == "rife4.14":
            from .rife414.RIFE_HDv3 import Model

            modelDir = os.path.dirname(
                os.path.join(
                    f"{thisdir}", "models", "rife-cuda", "rife414", "rife4.14.pkl"
                )
            )

        if self.interpolate_method == "rife4.6":
            from .rife46.RIFE_HDv3 import Model

            modelDir = os.path.dirname(
                os.path.join(
                    f"{thisdir}", "models", "rife-cuda", "rife46", "rife4.6.pkl"
                )
            )
        if self.interpolate_method == "rife4.13-lite":
            from .rife413lite.RIFE_HDv3 import Model

            modelDir = os.path.dirname(
                os.path.join(
                    f"{thisdir}",
                    "models",
                    "rife-cuda",
                    "rife413-lite",
                    "rife4.13-lite.pkl",
                )
            )
        if self.interpolate_method == "rife4.14-lite":
            from .rife414lite.RIFE_HDv3 import Model

            modelDir = os.path.dirname(
                os.path.join(
                    f"{thisdir}",
                    "models",
                    "rife-cuda",
                    "rife414-lite",
                    "rife4.14-lite.pkl",
                )
            )

        if self.interpolate_method == "rife4.15":
            from .rife415.RIFE_HDv3 import Model

            modelDir = os.path.dirname(
                os.path.join(
                    f"{thisdir}", "models", "rife-cuda", "rife415", "rife4.15.pkl"
                )
            )
        if self.interpolate_method == "rife4.16-lite":
            from .rife416lite.RIFE_HDv3 import Model

            modelDir = os.path.dirname(
                os.path.join(
                    f"{thisdir}",
                    "models",
                    "rife-cuda",
                    "rife416-lite",
                    "rife4.16-lite.pkl",
                )
            )
        # Apparently this can improve performance slightly
        torch.set_float32_matmul_precision("medium")

        if self.UHD:
            self.scale = 0.5

        ph = ((self.height - 1) // 64 + 1) * 64
        pw = ((self.width - 1) // 64 + 1) * 64
        self.padding = (0, pw - self.width, 0, ph - self.height)

        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_available else "cpu")

        torch.set_grad_enabled(False)
        if self.cuda_available:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half and not self.UHD:
                torch.set_default_dtype(torch.float16)

        self.model = Model(scale=self.scale, ensemble=self.ensemble)
        self.model.load_model(modelDir, -1)
        self.model.eval()

        if self.cuda_available and self.half and not self.UHD:
            self.model.half()

        self.model.device()
        self.I0 = None

    @torch.inference_mode()
    def make_inference(self, n):
        timestep = torch.full(
            (1, 1, self.I1.shape[2], self.I1.shape[3]), n, device=self.device
        ).to(memory_format=torch.channels_last)
        if self.half:
            timestep = timestep.half()
        output = self.model.inference(self.I0, self.I1, timestep=timestep)
        output = output[:, :, : self.height, : self.width]
        output = (
            (output[0])
            .squeeze(0)
            .permute(1, 2, 0)
            .mul(255.0)
            .byte()
            .contiguous()
            .cpu()
            .numpy()
        )
        self.cacheFrame()
        return output
    def clearCache(self):
        """
        Clears cache when scene change is detected.
        
        Overwrites frame
        """
        
        self.I0 = None

    def cacheFrame(self):
        self.I0 = self.I1.clone()
        
    @torch.inference_mode()
    def pad_frame(self,frame):
        return F.pad(frame, [0, self.padding[1], 0, self.padding[3]])

    @torch.inference_mode()
    def run(self, I1):
        if self.I0 is None:
            self.I0 = self.bytesToPaddedFrame(I1)
            print("Uncached frame")
            return False
        self.I1 = self.bytesToPaddedFrame(I1)
        return True
    
    @torch.inference_mode()
    def bytesToPaddedFrame(self, frame):
        

        if self.half:
            frame =  (
            torch.frombuffer(frame, dtype=torch.uint8)
            .reshape(self.height, self.width, 3)
            .to(self.device, non_blocking=True)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .half()
            .mul_(1 / 255)
            )
        #f32 
        else:
            frame = (
            torch.frombuffer(frame, dtype=torch.uint8)
            .reshape(self.height, self.width, 3)
            .to(self.device, non_blocking=True)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .mul_(1 / 255)
        )
        if self.padding != (0, 0, 0, 0):
            frame = self.pad_frame(frame)
    
        return frame