import os
import warnings
import platform
import numpy as np
import cv2

def isFlatpak():
    return "FLATPAK_ID" in os.environ


if isFlatpak():
    cwd = os.path.join(
        os.path.expanduser("~"), ".var", "app", "io.github.tntwise.REAL-Video-Enhancer"
    )
    if not os.path.exists(cwd):
        cwd = os.path.join(
            os.path.expanduser("~"),
            ".var",
            "app",
            "io.github.tntwise.REAL-Video-EnhancerV2",
        )
else:
    cwd = os.getcwd()
with open(os.path.join(cwd, "backend_log.txt"), "w") as f:
    pass


def warnAndLog(message: str):
    warnings.warn(message)
    log("WARN: " + message)


def errorAndLog(message: str):
    log("ERROR: " + message)
    raise os.error("ERROR: " + message)


def printAndLog(message: str, separate=False):
    """
    Prints and logs a message to the log file
    separate, if True, activates the divider
    """
    if separate:
        message = message + "\n" + "---------------------"
    print(message)
    log(message=message)


def log(message: str):
    with open(os.path.join(cwd, "backend_log.txt"), "a") as f:
        f.write(message + "\n")

def bytesTo100x100img(image: bytes, width, height) -> np.ndarray:
        frame = np.frombuffer(image,dtype=np.uint8).reshape(height, width, 3)
        frame = cv2.resize(
                frame, dsize=(100, 100)
            )
        return frame

def currentDirectory():
    return cwd

def ffmpegPath() -> str:
    return str(os.path.join(currentDirectory(),'bin','ffmpeg'))

def modelsDirectory():
    return os.path.join(cwd, "models")


def checkForPytorch() -> bool:
    """
    function that checks if the pytorch backend is available
    """
    try:
        import torch
        import torchvision
        import spandrel

        return True
    except ImportError as e:
        log(str(e))
        return False
    except Exception as e:
        log(str(e))


def checkForTensorRT() -> bool:
    """
    function that checks if the pytorch backend is available
    """
    try:
        import torch
        import torchvision
        import spandrel
        import tensorrt
        import torch_tensorrt

        return True
    except ImportError as e:
        log(str(e))
        return False
    except Exception as e:
        log(str(e))


def check_bfloat16_support() -> bool:
    """
    Function that checks if the torch backend supports bfloat16
    """
    import torch

    try:
        x = torch.tensor([1.0], dtype=torch.bfloat16)
        return True
    except RuntimeError:
        return False


def checkForDirectMLHalfPrecisionSupport() -> bool:
    """
    Function that checks if the onnxruntime DirectML backend supports half precision
    """
    try:
        import onnxruntime as ort
        import numpy as np
        import onnx
        from onnx import helper, TensorProto

        # Check if DirectML execution provider is available
        providers = ort.get_available_providers()
        if "DmlExecutionProvider" in providers:
            # Create a dummy model with half precision input
            input_shape = [1, 3, 224, 224]  # Example input shape
            input_tensor = helper.make_tensor_value_info(
                "input", TensorProto.FLOAT16, input_shape
            )
            output_tensor = helper.make_tensor_value_info(
                "output", TensorProto.FLOAT16, input_shape
            )
            node = helper.make_node("Identity", ["input"], ["output"])
            graph = helper.make_graph([node], "test", [input_tensor], [output_tensor])

            # Create the model
            model = helper.make_model(graph, producer_name="test_model")

            # Add opset version
            opset = helper.make_operatorsetid(
                "", 13
            )  # Use opset version 13 or any other appropriate version
            model.opset_import.extend([opset])

            # Set the IR version
            model.ir_version = onnx.IR_VERSION

            # Create an inference session with DirectML execution provider
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            session_options.add_session_config_entry("session.use_dml", "1")
            session = ort.InferenceSession(model.SerializeToString(), session_options)

            # Check if the model can be run with half precision input
            input_data = np.random.randn(*input_shape).astype(np.float16)
            outputs = session.run(None, {"input": input_data})

            return True
        else:
            log("DirectML execution provider not available")
            return False
    except ImportError as e:
        log(str(e))
        return False
    except Exception as e:
        log(str(e))
        return False


def checkForDirectML() -> bool:
    """
    Function that checks if the onnxruntime DirectML backend is available
    """
    try:
        import onnxruntime as ort
        import onnx
        import onnxconverter_common

        # Check if DirectML execution provider is available
        providers = ort.get_available_providers()
        if "DmlExecutionProvider" in providers:
            return True
        else:
            log("DirectML execution provider not available")
            return False
    except ImportError as e:
        log(str(e))
        return False
    except Exception as e:
        log(str(e))
        return False


def checkForNCNN() -> bool:
    """
    function that checks if the pytorch backend is available
    """
    try:
        from rife_ncnn_vulkan_python import Rife
        from upscale_ncnn_py import UPSCALE

        return True
    except ImportError as e:
        log(str(e))
        return False
    except Exception as e:
        log(str(e))
