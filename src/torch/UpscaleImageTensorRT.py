import os
try:
    import torch
except:
    pass
import numpy as np
import logging
# import torch_tensorrt as trt


from polygraphy.backend.trt import (
    TrtRunner,
    engine_from_network,
    network_from_onnx_path,
    CreateConfig,
    Profile,
    EngineFromBytes,
    SaveEngine,
)
from polygraphy.backend.common import BytesFromPath

# Apparently this can improve performance slightly
torch.set_float32_matmul_precision("medium")

class UpscaleTensorRT:
    def __init__(
        self,
        modelPath: str = "shufflecugan-tensorrt",
        upscaleFactor: int = 2,
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
        customModel: str = None,
        nt: int = 1,
    ):
        """
        Initialize the upscaler with the desired model

        Args:
            modelPath (str): The method to use for upscaling
            upscaleFactor (int): The factor to upscale by
            half (bool): Whether to use half precision
            width (int): The width of the input frame
            height (int): The height of the input frame
            customModel (str): The path to a custom model file
            nt (int): The number of threads to use
        """
        self.upscaleMethod = modelPath
        self.upscaleFactor = upscaleFactor
        self.half = half
        self.width = width
        self.height = height
        self.customModel = customModel
        self.nt = nt

        self.handleModel()

    def handleModel(self):
        # Reusing the directML models for TensorRT since both require ONNX models
        

        
        
        modelPath = '/home/pax/Downloads/2x_ModernSpanimationV1_fp16_op17.onnx'
        

        self.isCudaAvailable = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.isCudaAvailable else "cpu")
        if self.isCudaAvailable:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_dtype(torch.float16)

        # TO:DO account for FP16/FP32
        if not os.path.exists(modelPath.replace(".onnx", ".engine")):
            toPrint = f"Model engine not found, creating engine for model: {modelPath}, this may take a while..."
            print((toPrint))
            logging.info(toPrint)
            profiles = [
                # The low-latency case. For best performance, min == opt == max.
                Profile().add(
                    "input",
                    min=(1, 3, 8, 8),
                    opt=(1, 3, self.height, self.width),
                    max=(1, 3, 1080, 1920),
                ),
            ]
            self.engine = engine_from_network(
                network_from_onnx_path(modelPath),
                config=CreateConfig(fp16=self.half, profiles=profiles),
            )
            self.engine = SaveEngine(self.engine, modelPath.replace(".onnx", ".engine"))

        else:
            self.engine = EngineFromBytes(
                BytesFromPath(modelPath.replace(".onnx", ".engine"))
            )

        self.runner = TrtRunner(self.engine)
        self.runner.activate()

    @torch.inference_mode()
    def UpscaleImage(self, frame: np.ndarray) -> np.ndarray:
        frame = (
            torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().mul_(1 / 255)
        )

        return (
            self.runner.infer(
                {
                    "input": frame.half()
                    if self.half and self.isCudaAvailable
                    else frame
                },
                check_inputs=False,
            )["output"]
            .squeeze(0)
            .permute(1, 2, 0)
            .mul_(255)
            .byte()
            .cpu()
            .numpy()
        )

