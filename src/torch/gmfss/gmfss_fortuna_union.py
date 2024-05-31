import os
import torch
import numpy as np
from src.programData.thisdir import thisdir
thisdir = thisdir()
from torch.nn import functional as F

# from: https://github.com/HolyWu/vs-gmfss_fortuna/blob/master/vsgmfss_fortuna/__init__.py
class GMFSS():
    def __init__(self, interpolation_factor, half:bool, width:int, height:int, ensemble:bool,UHD:bool=False):

        self.width = width
        self.height = height
        self.half = half
        self.interpolation_factor = interpolation_factor
        self.UHD = UHD
        self.ensemble = ensemble
        # Yoinked from rife, needs further testing if these are the optimal
        # FLownet, from what I recall needs 32 paddings
        ph = ((self.height - 1) // 32 + 1) * 32
        pw = ((self.width - 1) // 32 + 1) * 32
        self.padding = (0, pw - self.width, 0, ph - self.height)

        if self.UHD == True:
            self.scale = 0.5
        else:
            self.scale = 1.0

        self.handle_model()

    def handle_model(self):

        # Apparently this can improve performance slightly
        torch.set_float32_matmul_precision("medium")
        
        
        model_type = "union"
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Doing a torch cuda check is rather expensive on start-up times so I just decided to keep it simple
        self.cuda_available = False
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            self.cuda_available = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_dtype(torch.float16)

        from .model.GMFSS import GMFSS as Model

        self.model = Model(os.path.join(thisdir,"models","gmfss-cuda"), model_type, self.scale, ensemble=False)
        self.model.eval().to(self.device, memory_format=torch.channels_last)

        self.dtype = torch.float
        if self.cuda_available:
            if self.half:
                self.model.half()
                self.dtype = torch.float16
                
              
    @torch.inference_mode()
    def make_inference(self, n):
        
        timestep = torch.tensor((n) * 1. , dtype=self.dtype, device=self.device)
        output = self.model(self.I0, self.I1, timestep)
        if self.padding != (0, 0, 0, 0):
            output = output[..., : self.height, : self.width]
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
        
        return output

    @torch.inference_mode()
    def pad_image(self, img):
        img = F.pad(img, self.padding)
        return img
    def bytesToFrame(self, frame):
        return (
            torch.frombuffer(frame, dtype=torch.uint8)
            .reshape(self.height, self.width, 3)
            .to(self.device, non_blocking=True)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .mul_(1 / 255)
        )
    def pad_frame(self):
        self.I0 = F.pad(self.I0, [0, self.padding[1], 0, self.padding[3]])
        self.I1 = F.pad(self.I1, [0, self.padding[1], 0, self.padding[3]])
    @torch.inference_mode()
    def run1(self, I0, I1):
        self.I0 = self.bytesToFrame(I0)
        self.I1 = self.bytesToFrame(I1)

        if self.cuda_available and self.half and not self.UHD:
            self.I0 = self.I0.half()
            self.I1 = self.I1.half()

        if self.padding != (0, 0, 0, 0):
            self.pad_frame()