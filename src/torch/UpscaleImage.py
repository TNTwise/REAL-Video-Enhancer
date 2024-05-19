try:
    from src.programData.thisdir import thisdir

    thisdir = thisdir()
except:
    thisdir = "/home/pax/.local/share/REAL-Video-Enhancer/"
try:
    import torch as torch
    from torch.nn import functional as F
    from spandrel import (
        ImageModelDescriptor,
        ModelDescriptor,
        ModelLoader,
        MaskedImageModelDescriptor,
    )

except:
    pass
import numpy as np
from  src.torch.inputToTorch import bytesToTensor
# from realsr_ncnn_vulkan_python import *
# from realsr_ncnn_vulkan_python import *


class UpscaleCUDA:
    @torch.inference_mode()
    def __init__(self, width, height, model, half):
        self.width = width
        self.height = height

        self.model = ModelLoader().load_from_file(model)
        assert isinstance(self.model, ImageModelDescriptor)

        self.isCudaAvailable = torch.cuda.is_available()
        self.half = half
        self.bf16 = False

        self.device = torch.device("cuda" if self.isCudaAvailable else "cpu")
        self.model = (
            self.model.eval().cuda() if self.isCudaAvailable else self.model.eval()
        )

        if self.isCudaAvailable:
            # self.stream = [torch.cuda.Stream() for _ in range(self.nt)]
            # self.currentStream = 0
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            try:
                if self.half:
                    torch.set_default_dtype(torch.float16)
                    self.model.half()
                    self.bf16 = False
            except:
                torch.set_default_dtype(torch.bfloat16)
                self.model.bfloat16()
                self.bf16 = True

    @torch.inference_mode()
    def UpscaleImage(self, frame):

        frame = bytesToTensor(frame=frame,
                      height=self.height,
                      width=self.width,
                      half=self.half,
                      bf16=self.bf16)
       
        if self.isCudaAvailable:
            # torch.cuda.set_stream(self.stream[self.currentStream])
            frame = frame.cuda(non_blocking=True)
            if self.half and not self.bf16:
                frame = frame.half()
            if self.bf16:
                frame = frame.bfloat16()
        frame = frame.contiguous(memory_format=torch.channels_last)

        output = self.model(frame)
        

        return (output).squeeze(0).permute(1, 2, 0).mul(255.0).byte().contiguous().cpu().numpy()
