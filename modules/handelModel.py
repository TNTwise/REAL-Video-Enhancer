from src.programData.thisdir import thisdir
import os

thisdir = thisdir()


def handleModel(AI, customModel=None):
    if AI == "realesrgan-cuda" or AI == "realesrgan-cuda-tensorrt":
        return os.path.join(
            f"{thisdir}", "models", "realesrgan-cuda", "realesr-animevideov3.pth"
        )
    if AI == "rife-cuda":  # handle rife models in rife file
        return None
    if AI == "custom-models-cuda" or AI == "custom-models-cuda-tensorrt":
        return os.path.join(
            f"{thisdir}", "models", f"custom-models-cuda", f"{customModel}"
        )
