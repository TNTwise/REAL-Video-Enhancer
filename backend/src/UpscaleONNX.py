import onnxruntime as ort
from onnxruntime import InferenceSession 
import os
import numpy as np

def getONNXScale(modelPath: str = "") -> int:
    paramName = os.path.basename(modelPath).lower()
    for i in range(100):
        if f"{i}x" in paramName or f"x{i}" in paramName:
            return i


class UpscaleONNX:
    def __init__(self, modelPath,deviceID: int = 0):
        self.modelPath = modelPath
        self.deviceID = deviceID
        self.inferenceSession = self.loadInferenceSession()
    
    def getScale(self) -> int:
        self.scale = getONNXScale(self.modelPath)
        return self.scale
   
    def bytesToFrame(self, image: bytes) -> tuple:
        image = np.frombuffer(image, dtype=np.uint8).reshape(1080, 1920, 3)
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image = image.__mul__(1.0 / 255.0)
        image = np.clip(image, 0, 1)
        return image

    def renderTensor(self, image_as_np_array: np.ndarray) -> np.ndarray:
        onnx_input  = {self.inferenceSession.get_inputs()[0].name: image_as_np_array}
        onnx_output = self.inferenceSession.run(None, onnx_input)[0]
        return self.frameToBytes(onnx_output)
    
    def frameToBytes(self, image: np.ndarray) -> bytes:
        image = np.clip(image, 0, 1)
        image = np.transpose(image, (0, 2, 3, 1))
        image = np.squeeze(image)
        image = image.__mul__(255.0)
        image = image.astype(np.uint8)
        image = image.tobytes()
        return image

    def loadInferenceSession(self) -> InferenceSession:        
        directml_backend = [('DmlExecutionProvider', {"device_id": f"{  self.deviceID }"})]

        inference_session = InferenceSession(path_or_bytes = self.modelPath, providers = directml_backend)

        return inference_session


if __name__ == "__main__":
    UpscaleONNX("2x_ModernSpanimationV1.pth_op17_torch.float32.onnx").setProvider()
