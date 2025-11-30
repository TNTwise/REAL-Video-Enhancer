import torch
import torch.nn.functional as F
class InferenceSceneChangeDetectEfficientNet:
    def __init__(self):
        self.model = torch.jit.load("/home/pax/real-video-enhancer/backend/src/pytorch/scenechangedetect/sudo_efficientnet_scenedetect.pt", map_location='cpu').float()
    
    @torch.inference_mode()
    def __call__(self, frame_0: torch.Tensor, frame_1: torch.Tensor) -> bool:
        frame_0 = frame_0.permute(2, 0, 1)  #shape: (3, H, W)
        frame_1 = frame_1.permute(2, 0, 1)
        #shape: (6, H, W)
        frame_0 = F.interpolate(frame_0.unsqueeze(0), 
                            size=(256, 256), 
                            mode='bilinear', 
                            )
        frame_1 = F.interpolate(frame_1.unsqueeze(0),
                            size=(256, 256), 
                            mode='bilinear', 
                            )
        #shape: (1, 3, 256, 256)
        
        input_tensor = torch.cat((frame_0.squeeze(0), frame_1.squeeze(0)), dim=0)
        #shape: (6, 256, 256)
        input_tensor = input_tensor.to(dtype=torch.float32, device='cpu')
        output = self.model(input_tensor)
        # Return True if scene change detected, else False
        return output[0][0] >= 0.85