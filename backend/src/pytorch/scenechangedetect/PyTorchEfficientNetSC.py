import torch
import torch.nn.functional as F
class InferenceSceneChangeDetectEfficientNet:
    def __init__(self):
        self.model = torch.jit.load("/home/pax/real-video-enhancer/backend/src/pytorch/scenechangedetect/sudo_maxxvit_scenedetect.pt", map_location='cuda').half()
        self.model.eval()
    @torch.inference_mode()
    def __call__(self, frame_0: torch.Tensor, frame_1: torch.Tensor) -> bool:
        frame = torch.cat((frame_0, frame_1), dim=0)
        output = self.model(frame)
        # Return True if scene change detected, else False
        
        return output[0][0] > 0.3