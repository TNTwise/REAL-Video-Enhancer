import onnx
import onnxruntime as ort
from onnxruntime import InferenceSession
from onnxconverter_common import float16
import os
import numpy as np
from .Util import checkForDirectMLHalfPrecisionSupport


def getONNXScale(modelPath: str = "") -> int:
    paramName = os.path.basename(modelPath).lower()
    for i in range(100):
        if f"{i}x" in paramName or f"x{i}" in paramName:
            return i


class UpscaleONNX:
    def __init__(self,
                modelPath,
                deviceID: int = 0, 
                precision: str = "float32",
                width: int = 1920,
                height: int = 1080,
                ):
        self.width = width
        self.height = height
        self.modelPath = modelPath
        self.deviceID = deviceID
        self.i0 = None
        self.precision = self.handlePrecision(precision)
        self.model = self.loadModel()
        self.inferenceSession = self.loadInferenceSession()

    def getScale(self) -> int:
        self.scale = getONNXScale(self.modelPath)
        return self.scale

    def handlePrecision(self, precision):
        if precision == "auto":
            return np.float16 if checkForDirectMLHalfPrecisionSupport() else np.float32
        if precision == "float16":
            return np.float16
        if precision == "float32":
            return np.float32

    def bytesToFrame(self, image: bytes) -> tuple:
        image = np.frombuffer(image, dtype=np.uint8).reshape(1080, 1920, 3)
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = image.astype(self.precision)
        image = image.__mul__(1.0 / 255.0)
        return np.ascontiguousarray(image)

    def renderTensor(self, image_as_np_array: np.ndarray) -> np.ndarray:
        onnx_input = {self.inferenceSession.get_inputs()[0].name: image_as_np_array}
        onnx_output = self.inferenceSession.run(None, onnx_input)[0]
        return self.frameToBytes(onnx_output)

    def frameToBytes(self, image: np.ndarray) -> bytes:
        image = (
            image.clip(0, 1)
            .squeeze()
            .transpose(1, 2, 0)
            .__mul__(255.0)
            .astype(np.uint8)
        )
        return np.ascontiguousarray(image).tobytes()

    def loadModel(self):
        model = onnx.load(self.modelPath)
        if self.precision == np.float16:
            model = float16.convert_float_to_float16(model)
        return model

    def loadInferenceSession(self) -> InferenceSession:
        directml_backend = [
            ("DmlExecutionProvider", {"device_id": f"{  self.deviceID }"})
        ]

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        inference_session = InferenceSession(
            self.model.SerializeToString(), session_options, providers=directml_backend
        )

        return inference_session


if __name__ == "__main__":
    UpscaleONNX("2x_ModernSpanimationV1.pth_op17_torch.float32.onnx").setProvider()
