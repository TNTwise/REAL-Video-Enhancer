import torch
from ...utils.Util import log
from ..TorchUtils import TorchUtils
import os

class UpscaleModelWrapper:
    def __init__(self, model_path: torch.nn.Module, device: torch.device, precision: torch.dtype):
        self.__model_path = model_path
        self.__device = device
        self.__precision = precision
        self.load_model()
        self.set_precision(self.__precision)
        self.__test_model_precision()

    def set_precision(self, precision: torch.dtype):
        self.__precision = precision
        self.__model.to(self.__device, dtype=precision)

    def get_model(self):
        return self.__model
    
    def get_scale(self):
        return self.__scale

    def load_state_dict(self, state_dict):
        self.__model.load_state_dict(state_dict)

    def __test_inference(self, test_input:torch.Tensor):
        # inference and get re-load state dict due to issue with span.
        with torch.inference_mode():
            model = self.__model
            model(test_input)
            output = model(test_input)
            self.__model.load_state_dict(model.state_dict()) # reload state dict to fix span
            del model
            TorchUtils.clear_cache()

    def __test_model_precision(self):
        test_input = torch.randn(1, 3, 64, 64).to(self.__device, dtype=self.__precision)
        with torch.inference_mode():
            try:
                self.__test_inference(test_input)
            except Exception as e:
                log(f"Model precision {self.__precision} not supported, falling back to float32: {e}")
                self.set_precision(torch.float32)
                self.__test_inference(test_input)
            

    @torch.inference_mode()
    def load_model(self, model=None) -> torch.nn.Module:
        if not model:
            try:
                from . import ModelLoader, ImageModelDescriptor, UnsupportedModelError
            except ImportError:
                # spandrel will import like this if its a submodule
                from .libs.spandrel.spandrel import ModelLoader, ImageModelDescriptor, UnsupportedModelError
            try:
                model = ModelLoader().load_from_file(self.__model_path)
                assert isinstance(model, ImageModelDescriptor)
                # get model attributes
                
            except (UnsupportedModelError) as e:
                log(f"Model at {self.__model_path} is not supported: {e}")
                raise e
            
            self.__scale = model.scale
            model = model.model

        self.__model = model
    
    def __call__(self, *args, **kwargs):
        return self.__model(*args, **kwargs)