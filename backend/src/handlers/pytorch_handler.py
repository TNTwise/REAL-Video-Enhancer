class TorchHandler:
    def __init__(self):
        try:
            import torch
            import torchvision

            self.torch = torch
            self.torchvision = torchvision
        except ImportError as e:
            print(f"Error importing torch: {e}")
            self.torch = None
            self.torchvision = None

    def is_available(self):
        return self.torch is not None and self.torchvision is not None

    def get_torch(self):
        if not self.is_available():
            raise RuntimeError("PyTorch is not available")
        return self.torch

    def get_torchvision(self):
        if not self.is_available():
            raise RuntimeError("TorchVision is not available")
        return self.torchvision
