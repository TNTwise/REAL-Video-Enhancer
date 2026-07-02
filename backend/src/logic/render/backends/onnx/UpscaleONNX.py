import os
import pathlib
import time

import numpy as np
import onnxruntime as ort
from onnxconverter_common import float16
from onnxruntime import InferenceSession

import backend.src.proxy.backends.onnx as onnx


def getONNXScale(modelPath: str = "") -> int:
    paramName = os.path.basename(modelPath).lower()
    for i in range(100):
        if f"{i}x" in paramName or f"x{i}" in paramName:
            return i


class UpscaleONNX:
    @classmethod
    def getModelScale(cls, modelPath: str) -> int:
        model_name = os.path.basename(modelPath).lower()
        for i in range(100):
            if f"{i}x" in model_name or f"x{i}" in model_name:
                return i
        raise ValueError("Scale not found in model name!")

    def __init__(
        self,
        modelPath: str,
        device="default",
        tile_pad: int = 10,
        precision: str = "auto",
        width: int = 1920,
        height: int = 1080,
        scale: int = 2,
        tilesize: int = 0,
        gpu_id: int = 0,
        hdr_mode: bool = False,
    ):
        self.width = width
        self.height = height
        self.scale = scale
        self.modelPath = modelPath
        self.device = device
        self.i0 = None
        self.precision = np.float16

        self.input_buffer = np.empty(
            (1, 3, self.height, self.width), dtype=self.precision
        )
        self.output_buffer = np.empty(
            (1, self.height * self.scale, self.width * self.scale, 3),
            dtype=self.precision,
        )

        # Pre-compute normalization factor
        self.norm_factor = np.array(1.0 / 255.0, dtype=self.precision)

        # load model
        model = onnx.load(self.modelPath)

        if self.precision == np.float16:
            model = float16.convert_float_to_float16(model, check_fp16_ready=False)
            # Optimized DirectML provider options
        self.model = model
        directml_options = {
            "device_id": gpu_id,
            #    "enable_dynamic_graph_fusion": True,
            #    "disable_memory_arena": False,  # Keep memory arena for better performance
            #   "memory_limit_in_mb": 0,  # Use all available memory
        }

        directml_backend = [("DmlExecutionProvider", directml_options)]

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        # session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # session_options.enable_mem_pattern = True

        # Add these for better DirectML performance
        # session_options.add_session_config_entry("session.disable_prepacking", "0")
        # session_options.add_session_config_entry("session.enable_memory_efficient_execution", "1")

        self.inference_session = InferenceSession(
            self.model.SerializeToString(),
            session_options,
            providers=directml_backend,
        )

    def bytesToFrame(self, image: bytes) -> tuple:
        temp_view = np.frombuffer(image, dtype=np.uint8)
        temp_view = temp_view.reshape(self.height, self.width, 3)

        # Use direct assignment to pre-allocated buffer
        self.input_buffer[0] = np.transpose(temp_view, (2, 0, 1))

        # In-place normalization
        self.input_buffer *= self.norm_factor

        return self.input_buffer
        image = np.frombuffer(image, dtype=np.uint8).reshape(1080, 1920, 3)
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = image.astype(self.precision)
        image = image.__mul__(1.0 / 255.0)
        return np.ascontiguousarray(image)

    def renderTensor(self, image_as_np_array: np.ndarray) -> np.ndarray:
        onnx_input = {self.inference_session.get_inputs()[0].name: image_as_np_array}
        onnx_output = self.inference_session.run(None, onnx_input)[0]
        return onnx_output

    def frameToBytes(self, image: np.ndarray) -> bytes:
        self.output_buffer[0] = image.clip(0, 1).squeeze().transpose(1, 2, 0)
        self.output_buffer[0] /= self.norm_factor
        image = (
            self.output_buffer[0]
            .astype(np.uint8)
            .reshape(self.height * self.scale, self.width * self.scale, 3)
        )
        return np.ascontiguousarray(image).tobytes()

    def hotUnload(self):
        self.paused = True

    def hotReload(self):
        self.paused = False

    def __call__(self, image: bytes) -> bytes:
        while self.paused:
            time.sleep(1)
        image_as_np_array = self.bytesToFrame(image)
        output = self.renderTensor(image_as_np_array)
        return self.frameToBytes(output)


if __name__ == "__main__":

    def download_file(url, local_path):
        import requests

        response = requests.get(url, stream=True)
        response.raise_for_status()  # Ensure we notice bad responses
        with pathlib.Path(local_path).open("wb") as f:
            f.writelines(response.iter_content(chunk_size=8192))

    if not pathlib.Path(
        "2x_ModernSpanimationV2_clamp_op20_fp16_onnxslim.onnx"
    ).is_file():
        download_file(
            "https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/2x_ModernSpanimationV2_clamp_op20_fp16_onnxslim.onnx",
            "2x_ModernSpanimationV2_clamp_op20_fp16_onnxslim.onnx",
        )
    if not pathlib.Path("models.png").is_file():
        download_file(
            "https://github.com/TNTwise/REAL-Video-Enhancer/blob/v2-main/screenshots/models.png?raw=true",
            "models.png",
        )

    up = UpscaleONNX("2x_ModernSpanimationV2_clamp_op20_fp16_onnxslim.onnx")
    image = cv2.imread("models.png")
    image = cv2.resize(image, (1920, 1080)).astype(np.uint8).tobytes()
    start_time = time.time()

    iter = 100
    # import viztracer
    # tracer = viztracer.VizTracer()
    # tracer.start()

    image1 = up.bytesToFrame(image)
    cv2.imwrite(
        "input.jpg", np.frombuffer(image, dtype=np.uint8).reshape(1080, 1920, 3)
    )
    output = up.renderTensor(image1)
    o = up.frameToBytes(output)
    end_time = time.time()

    # tracer.stop()
    # tracer.save("onnx_viztracer_result.json")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    fps = iter / (end_time - start_time)
    print(f"FPS: {fps:.2f}")

    output = up.frameToBytes(output)
    output = np.frombuffer(output, dtype=np.uint8).reshape(1080 * 2, 1920 * 2, 3)

    cv2.imwrite("output.jpg", output)
    print("Done")
