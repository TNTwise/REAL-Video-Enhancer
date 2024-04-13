import os
import torch
import cupy
from torch.nn import functional as F
try:
    import src.programData.thisdir

    thisdir = src.programData.thisdir.thisdir()
except:
    thisdir = f"{os.path.expanduser(r'~')}/.local/share/REAL-Video-Enhancer"
from src.misc.log import log
# from: https://github.com/HolyWu/vs-gmfss_fortuna/blob/master/vsgmfss_fortuna/__init__.py


class GMFSS:
    def __init__(
        self, interpolation_factor, half, width, height, UHD=False, ensemble=False, nt=1
    ):
        self.width = width
        self.height = height
        self.half = half
        self.interpolation_factor = interpolation_factor
        self.UHD = UHD
        self.ensemble = ensemble
        self.nt = nt

        ph = ((self.height - 1) // 32 + 1) * 32
        pw = ((self.width - 1) // 32 + 1) * 32
        self.padding = (0, pw - self.width, 0, ph - self.height)

        if self.UHD:
            self.scale = 0.5
        else:
            self.scale = 1.0

        self.handle_model()

    def handle_model(self):
        torch.set_float32_matmul_precision("medium")

        
        modelDir = os.path.join(f"{thisdir}","models","gmfss-cuda")

        modelType = "union"

        self.isCudaAvailable = torch.cuda.is_available()

        if not self.isCudaAvailable:
            print(
                "CUDA is not available, using CPU. Expect significant slowdows or no functionality at all. If you have a NVIDIA GPU, please install CUDA and make sure that CUDA_Path is in the environment variables. CUDA Installation link: https://developer.nvidia.com/cuda-downloads"
            )

            log(
                "CUDA is not available, using CPU. Expect significant slowdows or no functionality at all. If you have a NVIDIA GPU, please install CUDA and make sure that CUDA_Path is in the environment variables. CUDA Installation link: https://developer.nvidia.com/cuda-downloads"
            )
        try:
            cupy.cuda.get_cuda_path()
        except Exception:
            log(
                "Couldn't find relevant CUDA installation. Please make sure that CUDA_Path is in the environment variables. CUDA Installation link: https://developer.nvidia.com/cuda-downloads"
            )

            raise Exception(
                "Couldn't find relevant CUDA installation. Please make sure that CUDA_Path is in the environment variables. CUDA Installation link: https://developer.nvidia.com/cuda-downloads"
            )

        self.device = torch.device("cuda" if self.isCudaAvailable else "cpu")

        torch.set_grad_enabled(False)
        if self.isCudaAvailable:
            self.stream = [torch.cuda.Stream() for _ in range(self.nt)]
            self.current_stream = 0
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)

        from .model.GMFSS import GMFSS as Model

        self.model = Model(modelDir, modelType, self.scale, ensemble=self.ensemble)
        self.model.eval().to(self.device, memory_format=torch.channels_last)

        self.dtype = torch.float
        if self.isCudaAvailable and self.half:
            self.model.half()
            self.dtype = torch.half

        self.I0 = None

    @torch.inference_mode()
    def make_inference(self, n):
        
        if self.isCudaAvailable:
            torch.cuda.set_stream(self.stream[self.current_stream])
        

        timestep = torch.tensor(
            (n + 1) * 1.0 / (self.interpolation_factor + 1),
            dtype=self.dtype,
            device=self.device,
        )
        output = self.model(self.I0, self.I1, timestep)
        output = (output[0] * 255.0).byte().cpu().numpy().transpose(1, 2, 0)

        if self.isCudaAvailable:
            torch.cuda.synchronize(self.stream[self.current_stream])
            self.current_stream = (self.current_stream + 1) % len(self.stream)

        return output[: self.height, : self.width, :]

    @torch.inference_mode()
    def pad_image(self, img):
        img = F.pad(img, self.padding)
        return img

    def cacheFrame(self):
        self.I0 = self.I1.clone()

    @torch.inference_mode()
    def processFrame(self, frame):
        frame = (
            torch.from_numpy(frame)
            .to(self.device, non_blocking=True)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .mul_(1 / 255)
        )
        
        if self.isCudaAvailable and self.half:
            frame = frame.half()

        if self.padding != (0, 0, 0, 0):
            frame = F.pad(frame, [0, self.padding[1], 0, self.padding[3]])

        return frame.contiguous(memory_format=torch.channels_last)
    def pad_frame(self):
        self.I0 = F.pad(self.I0, [0, self.padding[1], 0, self.padding[3]])
        self.I1 = F.pad(self.I1, [0, self.padding[1], 0, self.padding[3]])

    @torch.inference_mode()
    def run(self, I1):
        if self.I0 is None:
            self.I0 = self.processFrame(I1)
            return False

        self.I1 = self.processFrame(I1)
        return True
    @torch.inference_mode()
    def run1(self,I0,I1):
        self.I0 = (
            torch.from_numpy(I0)
            .to(self.device, non_blocking=True)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        self.I1 = (
            torch.from_numpy(I1)
            .to(self.device, non_blocking=True)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255.0
        )

        if self.half:
            self.I0 = self.I0.half()
            self.I1 = self.I1.half()

        if self.padding != (0, 0, 0, 0):
            self.pad_frame()
