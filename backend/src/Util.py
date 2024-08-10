import os
import warnings
import tarfile

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


def currentDirectory():
    return cwd


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
        printAndLog(str(e))
        return False


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
        printAndLog(str(e))
        return False


def checkForNCNN() -> bool:
    """
    function that checks if the pytorch backend is available
    """
    try:
        from rife_ncnn_vulkan_python import RIFE
        from upscale_ncnn_py import UPSCALE

        return True
    except ImportError as e:
        printAndLog(str(e))
        return False
