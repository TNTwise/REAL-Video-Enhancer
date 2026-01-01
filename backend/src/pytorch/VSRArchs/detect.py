import torch
class DetectionHelper:
    def __init__(self):
        self.model = None

    def get_inference_method(self, modelPath: str):
        self.model = torch.load(modelPath)
        try:
            from AnimeSR import animesr_arch, vsr_inference_helper
            self.model = animesr_arch.AnimeSR()
            self.model.load_state_dict(self.model)
            return self.model
        except Exception as e:
            pass
        try:
            from TSPAN import tspan, vsr_inference_helper
            self.model = tspan.TemporalSPAN(upscale=1)
            self.model.load_state_dict(self.model)
            return self.model
        except Exception as e:
            pass

    @torch.inference_mode()
    def inference(self, frame: torch.Tensor):
        if self.model is None:
            raise ValueError("Model not loaded. Please call get_inference_method first.")
        with torch.no_grad():
            output = self.model(frame)
        return output
            
