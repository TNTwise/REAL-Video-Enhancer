import os
import torch
import torch.nn.functional as F
from ..TorchUtils import TorchUtils

class InferenceSceneChangeDetectEfficientNet:
    """
    Docstring for InferenceSceneChangeDetectEfficientNet
    """
    def __init__(self, 
                 threshold=0.3, 
                 model_path="",
                 model_dtype="float32", 
                 model_device="cpu", 
                 model_backend="pytorch",
                 ):
        self.threshold = 1 - threshold * .1
        model_dtype = TorchUtils.handle_precision(model_dtype)
        model_device = TorchUtils.handle_device(model_device)
        self.model_path = model_path
        self.device = model_device
        self.dtype = model_dtype
        exported = torch.export.load(model_path)
        scripted_model = exported.module().to(device=model_device, dtype=model_dtype)
        self.model = scripted_model
            
        
        if model_backend == "helpme":
            from ..TensorRTHandler import TorchTensorRTHandler, torchscript_to_dynamo
            dummy_input = torch.randn(6, 256, 256).to(device=model_device, dtype=model_dtype)
            trtHandler = TorchTensorRTHandler(os.path.dirname(self.model_path))

            trt_engine_name = os.path.join(
                    (
                        f"{os.path.basename(self.model_path)}"
                        + f"_{'fp16' if self.dtype == torch.float16 else 'fp32'}"
                        + f"_{torch.cuda.get_device_name(self.device)}"
                        + f"_trt-{trtHandler.tensorrt_version}"
                        + f"_torch_tensorrt-{trtHandler.torch_tensorrt_version}"
                        + f"_opt-3"
                        + ".trt"
                    ),
                )
            
            if os.path.exists(trt_engine_name):
                self.model = trtHandler.load_engine(trt_engine_name)
            
            else:
                self.model = trtHandler.build_engine(
                    scripted_model, 
                    device=model_device, 
                    dtype=model_dtype,
                    example_inputs=dummy_input,
                    trt_engine_name=trt_engine_name,
                    )
                
                trtHandler.save_engine(trt_engine_name, self.model, [dummy_input])
        
    """
    InferenceSceneChangeDetectEfficientNet class for detecting scene changes using an EfficientNet model.
    Args:
    threshold (float): The threshold value for detecting scene changes.
    model_path (str): The path to the trained model.
    model_dtype (str): The data type of the model (e.g., "float3
    "float16", etc.).
    model_device (str): The device to run the model on (e.g., "cpu
    "cuda", etc.).
    model_backend (str): The backend to use for the model (e.g., "script
    "trace", etc.).
    """
    @torch.inference_mode()
    def __call__(self, frame_0: torch.Tensor, frame_1: torch.Tensor) -> bool:
        # frame format: (C, H, W), values in [0, 1] or [0, 255]
        frame = torch.cat((frame_0, frame_1), dim=0)
        #inference format: (6, H, W)
        output = self.model(frame)
        # Return True if scene change detected, else False
        return output[0][0] > self.threshold 