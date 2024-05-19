import os
try:
    import torch
except:
    pass
import numpy as np
import onnx
import onnxruntime
from src.programData.thisdir import thisdir as th
thisdir = th()
# import torch_tensorrt as trt
try:
    import tensorrt
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
    from spandrel import ModelLoader
except Exception as e:
    print(e)
from src.misc.log import log
from  src.torch.inputToTorch import bytesToTensor

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
        modelName: str = None,
        nt: int = 1,
        guiLog = None,
    ):
        """
        Initialize the upscaler with the desired model

        Args:
            modelPath (str): The method to use for upscaling
            upscaleFactor (int): The factor to upscale by
            half (bool): Whether to use half precision
            width (int): The width of the input frame
            height (int): The height of the input frame
            nt (int): The number of threads to use
        """
        self.modelPath = modelPath
        self.upscaleFactor = upscaleFactor
        self.half = half
        self.width = width
        self.height = height
        self.modelName = modelName
        self.nt = nt
        self.bf16 = False
        self.onnxModelsPath = os.path.join(f"{thisdir}", "models", "onnx-models")
        self.locationOfOnnxModel = os.path.join(f'{self.onnxModelsPath}',f'{modelName}.onnx')
        self.guiLog = guiLog
        if not os.path.exists(self.locationOfOnnxModel):
            self.pytorchExportToONNX()
        self.handleModel()

    def pytorchExportToONNX(self): # Loads model via spandrel, and exports to onnx
        model = ModelLoader().load_from_file(self.modelPath)
        model = model.model
        state_dict = model.state_dict()
        model.eval().cuda()
        model.load_state_dict(state_dict, strict=True)
        input = torch.rand(1, 3, 256, 256).cuda()
        if self.half:
            try:
                
                    model.half()
                    input = input.half()
            except:
                model.bfloat16()
                input.bfloat16()
                self.bf16 = True
        self.guiLog.emit("Exporting ONNX")
        with torch.inference_mode():
            
           
            torch.onnx.export(
                model,
                input,
                self.locationOfOnnxModel,
                verbose=False,
                opset_version=17,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                        "input": {0: "batch_size", 2: "width", 3: "height"},
                        "output": {0: "batch_size", 2: "width", 3: "height"},
                    }
            )
    def handleModel(self):
        # Reusing the directML models for TensorRT since both require ONNX models
        

        
        
        
        

        self.isCudaAvailable = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.isCudaAvailable else "cpu")
        self.trt_version = tensorrt.__version__
        self.device_name = torch.cuda.get_device_name(self.device)
        if self.isCudaAvailable:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_dtype(torch.float16)

        # TO:DO account for FP16/FP32
        self.enginePath = f'{self.locationOfOnnxModel.replace(".onnx", "")}_{self.width}x{self.height}_half={self.half}_tensorrtVer={self.trt_version}device={self.device_name}_bf16={self.bf16}.engine'
        if not os.path.exists(self.enginePath):
            toPrint = f"Model engine not found, creating engine for model: {self.locationOfOnnxModel}, this may take a while..."
            self.guiLog.emit("Building Engine, this may take a while...")
            print((toPrint))
            profiles = [
                # The low-latency case. For best performance, min == opt == max.
                Profile().add(
                    "input",
                    min=(1, 3, self.height, self.width),
                    opt=(1, 3, self.height, self.width),
                    max=(1, 3, self.height, self.width),
                ),
            ]
            self.engine = engine_from_network(
                network_from_onnx_path(self.locationOfOnnxModel),
                config=CreateConfig(fp16=self.half, profiles=profiles),
            )
            self.engine = SaveEngine(self.engine, self.enginePath)

        else:
            self.engine = EngineFromBytes(
                BytesFromPath(self.enginePath)
            )

        self.runner = TrtRunner(self.engine)
        self.runner.activate()

    @torch.inference_mode()
    def UpscaleImage(self, frame: bytearray):
        frame = bytesToTensor(frame=frame,
                      height=self.height,
                      width=self.width,
                      half=self.half,
                      bf16=self.bf16)
        
        return (
            self.runner.infer(
                {
                    "input": frame
                    
                    
                },
                check_inputs=False,
            )["output"]
            .squeeze(0)
            .permute(1, 2, 0)
            .mul_(255)
            .byte()
            .contiguous()
            .cpu()
            .numpy()
        )

