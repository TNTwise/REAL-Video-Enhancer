from spandrel import ImageModelDescriptor, ModelLoader
from src.programData.thisdir import thisdir
import torch as torch
from torch.nn import functional as F

thisdir = thisdir()


class UpscaleCUDA:
    def __init__(self,
                 width,
                 height):
        self.width = width
        self.height = height
        self.model = ModelLoader().load_from_file(f"{thisdir}/models/realesrgan-cuda/realesr-animevideov3.pth")

        assert isinstance(self.model, ImageModelDescriptor)
        
        self.model.eval() # gonna have to put cuda back in here lmfaooooooo
        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_available else "cpu")
        
    def pad_frame(self,image):
        return F.pad(image, [0, self.padding[1], 0, self.padding[3]])
        
        
    def UpscaleImage(self,npArr):
        image = (
            torch.from_numpy(npArr)
            .to(self.device, non_blocking=True)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        
        ph = ((self.height - 1) // 64 + 1) * 64
        pw = ((self.width - 1) // 64 + 1) * 64
        self.padding = (0, pw - self.width, 0, ph - self.height)
        
        if self.padding != (0, 0, 0, 0):
            image = self.pad_frame(image)
            
        
        output = self.model(image)
        output = (output[0] * 255.0).byte().cpu().numpy().transpose(1, 2, 0)
        return output