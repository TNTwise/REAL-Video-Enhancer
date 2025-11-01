import torch
import gc
import sys
from ...TorchUtils import TorchUtils
class TemporalSPANInferenceHelper:
    def __init__(self, model, scale):
        self.scale = scale
        self.model: torch.nn.Module = model
        self.num_cached_frames = 5
        self.frame_cache = []
    
    def state_dict(self):
        self.frame_cache = [] # reset cache before saving state
        return self.model.state_dict()

    def __call__(self, frame: torch.Tensor):
        if len(self.frame_cache) == 0:
            for i in range(self.num_cached_frames):
                self.frame_cache.append(frame.unsqueeze(1))
        x = torch.cat(self.frame_cache, dim=1)
        out = self.model(x)
        self.frame_cache.pop(0)
        self.frame_cache.append(frame.unsqueeze(1))
        return out

            