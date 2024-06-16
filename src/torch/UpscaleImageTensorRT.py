import os

try:
    import torch
except:
    pass
import numpy as np
import onnx
from src.programData.thisdir import thisdir as th

thisdir = th()
import sys
import site
# import torch_tensorrt as trt

from src.misc.log import log

try:
    from src.torch.inputToTorch import bytesToTensor
    import tensorrt as trt
    from spandrel import ModelLoader
except:
    pass
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
        guiLog=None,
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

        self.TrtRunner = TrtRunner
        self.engine_from_network = engine_from_network
        self.network_from_onnx_path = network_from_onnx_path
        self.CreateConfig = CreateConfig
        self.Profile = Profile
        self.EngineFromBytes = EngineFromBytes
        self.SaveEngine = SaveEngine

        self.modelPath = modelPath
        self.upscaleFactor = upscaleFactor
        self.half = half
        self.width = width
        self.height = height
        self.modelName = modelName
        self.nt = nt
        self.bf16 = False
        self.onnxOpsetVersion = 18
        self.onnxModelsPath = os.path.join(f"{thisdir}", "models", "onnx-models")
        self.locationOfOnnxModel = os.path.join(
            f"{self.onnxModelsPath}",
            f"{modelName}-half={self.half}-scale{self.upscaleFactor}-opset{self.onnxOpsetVersion}.onnx",
        )
        self.guiLog = guiLog
        if not os.path.exists(self.locationOfOnnxModel):
            self.pytorchExportToONNX()
        self.handleModel()
        
    def handlePrecision(self):
        pass
    
    def pytorchExportToONNX(self):  # Loads model via spandrel, and exports to onnx
        model = ModelLoader().load_from_file(self.modelPath)
        model = model.model
        state_dict = model.state_dict()
        model.eval().cuda()
        model.load_state_dict(state_dict, strict=True)
        input = torch.rand(1, 3, 20, 20).cuda()
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
                opset_version=self.onnxOpsetVersion,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size", 2: "width", 3: "height"},
                    "output": {0: "batch_size", 2: "width", 3: "height"},
                },
            )

    def handleModel(self):
        # Reusing the directML models for TensorRT since both require ONNX models

        self.isCudaAvailable = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.isCudaAvailable else "cpu")
        self.trt_version = trt.__version__
        self.device_name = torch.cuda.get_device_name(self.device)
        if self.isCudaAvailable:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if self.half:
                torch.set_default_dtype(torch.float16)

        # TO:DO account for FP16/FP32
        self.enginePath = f'{self.locationOfOnnxModel.replace(".onnx", "")}{self.width}x{self.height}_scaleFactor={self.upscaleFactor}_half={self.half}_tensorrtVer={self.trt_version}device={self.device_name}_bf16={self.bf16}.engine'
        if not os.path.exists(self.enginePath):
            toPrint = f"Model engine not found, creating engine for model: {self.locationOfOnnxModel}, this may take a while..."
            self.guiLog.emit("Building Engine, this may take a while...")
            print((toPrint))
            profiles = [
                # The low-latency case. For best performance, min == opt == max.
                self.Profile().add(
                    "input",
                    min=(1, 3, self.height, self.width),
                    opt=(1, 3, self.height, self.width),
                    max=(1, 3, self.height, self.width),
                ),
            ]
            self.engine = self.engine_from_network(
                self.network_from_onnx_path(self.locationOfOnnxModel),
                config=self.CreateConfig(fp16=self.half, profiles=profiles),
            )
            self.engine = self.SaveEngine(self.engine, self.enginePath)

            with self.TrtRunner(self.engine) as runner:
                self.runner = runner

        with open(self.enginePath, "rb") as f, trt.Runtime(
            trt.Logger(trt.Logger.INFO)
        ) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()

        self.stream = torch.cuda.Stream()
        self.dummyInput = torch.zeros(
            (1, 3, self.height, self.width),
            device=self.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        self.dummyOutput = torch.zeros(
            (1, 3, self.height * self.upscaleFactor, self.width * self.upscaleFactor),
            device=self.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        self.bindings = [self.dummyInput.data_ptr(), self.dummyOutput.data_ptr()]

        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(
                self.engine.get_tensor_name(i), self.bindings[i]
            )
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensor_name, self.dummyInput.shape)

    @torch.inference_mode()
    def UpscaleImage(self, frame: bytearray):
        with torch.cuda.stream(self.stream):
            self.dummyInput.copy_(
                bytesToTensor(
                    frame,
                    half=self.half,
                    bf16=self.bf16,
                    width=self.width,
                    height=self.height,
                )
            )

            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            self.stream.synchronize()

            return (
                self.dummyOutput.squeeze(0)
                .permute(1, 2, 0)
                .mul_(255)
                .clamp(0, 255)
                .contiguous()
                .byte()
                .cpu()
                .numpy()
            )
        """frame = bytesToTensor(frame=frame,
                      height=self.height,
                      width=self.width,
                      half=self.half,
                      bf16=self.bf16)
        
        return (
            self.runner.infer(
                {
                    "input": frame.contiguous()
                    
                    
                },
                check_inputs=False,
            )["output"]
            .squeeze(0)
            .permute(1, 2, 0)
            .mul_(255)
            .clamp(0,255) # gotta clamp
            .contiguous()
            .byte()
            .cpu()
            .numpy()
        )"""
