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
        self.threshold = threshold * .1
        model_dtype = TorchUtils.handle_precision(model_dtype)
        model_device = TorchUtils.handle_device(model_device)
        self.model = torch.jit.load(model_path, map_location=model_device).to(dtype=model_dtype)
        self.model.eval()
            
        
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