import onnxruntime as ort
import torch
import cv2


class UpscaleONNX:
    def __init__(self, modelPath):
        self.modelPath = modelPath

    def setProvider(self):
        providers = [
            "CUDAExecutionProvider",
            "ROCMExecutionProvider",
            "CPUExecutionProvider",
        ]

        self.session = ort.InferenceSession(self.modelPath, providers=providers)
        self.session.run(None, {"input": torch.rand(1, 3, 256, 256).numpy()})
        outputs = self.session.run(None, {"input": torch.rand(1, 3, 256, 256).numpy()})

        cv2.imwrite(filename="out.png", img=outputs[0])


if __name__ == "__main__":
    UpscaleONNX("2x_ModernSpanimationV1.pth_op17_torch.float32.onnx").setProvider()
