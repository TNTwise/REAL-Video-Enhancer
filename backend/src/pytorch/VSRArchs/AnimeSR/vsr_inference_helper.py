import torch
import gc
from ...TorchUtils import TorchUtils
class AnimeSRInferenceHelper:
    def __init__(self, model, scale):
        self.scale = scale
        self.model: torch.nn.Module = model
        self.num_cached_frames = 3
        self.frame_cache = []
    
    def state_dict(self):
        self.frame_cache = [] # reset cache before saving state
        return self.model.state_dict()

    def __call__(self, frame: torch.Tensor):
        # fill the queue to render with multiple frames for the model
        TorchUtils.clear_cache()
        if len(self.frame_cache) == 0:
            height, width = frame.shape[2:]
            self.state = frame.new_zeros(1, 64, height, width)
            self.out = frame.new_zeros(1, 3, height * self.scale, width * self.scale)
            for i in range(self.num_cached_frames):
                self.frame_cache.append(frame)
        x = torch.cat(self.frame_cache, dim=1)
        
        #print(x.shape, file=sys.stderr)
        #sys.exit()
        self.out, self.state = self.model(x, self.out, self.state)
        # remove frame from cache
        self.frame_cache.pop(0)
        self.frame_cache.append(frame)
        #gc.collect()
        return self.out

            