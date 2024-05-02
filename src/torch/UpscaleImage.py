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
        
        self.device = torch.device("cuda" if self.isCudaAvailable else "cpu")
        self.model = (
            self.model.eval().cuda() if self.isCudaAvailable else self.model.eval()
        )

        
        if self.isCudaAvailable:
            # self.stream = [torch.cuda.Stream() for _ in range(self.nt)]
            # self.currentStream = 0
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_dtype(torch.float16)
                self.model.half()

    @torch.inference_mode()
    def UpscaleImage(self, frame):
        with torch.no_grad():
            frame = (
                torch.from_numpy(frame)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .mul_(1 / 255)
            )
            if self.isCudaAvailable:
                # torch.cuda.set_stream(self.stream[self.currentStream])
                frame = frame.cuda(non_blocking=True)
                if self.half:
                    frame = frame.half()
            frame = frame.contiguous(memory_format=torch.channels_last)

            output = self.model(frame)
            output = output.squeeze(0).permute(1, 2, 0).mul_(255).byte()

            return output.cpu().numpy()
