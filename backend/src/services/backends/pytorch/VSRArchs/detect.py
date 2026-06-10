import logging

import torch

logger = logging.getLogger(__name__)


class DetectionHelper:
    def __init__(self):
        self.model = None

    def get_inference_method(self, modelPath: str):
        self.model = torch.load(modelPath)
        try:
            from AnimeSR import animesr_arch

            self.model = animesr_arch.AnimeSR()
            self.model.load_state_dict(self.model)
            return self.model
        except Exception:
            logger.exception("Failed to load AnimeSR VSR architecture")
        try:
            from TSPAN import tspan

            self.model = tspan.TemporalSPAN(upscale=1)
            self.model.load_state_dict(self.model)
            return self.model
        except Exception:
            logger.exception("Failed to load TSPAN VSR architecture")

    @torch.inference_mode()
    def inference(self, frame: torch.Tensor):
        if self.model is None:
            raise ValueError(
                "Model not loaded. Please call get_inference_method first."
            )
        with torch.no_grad():
            output = self.model(frame)
        return output
