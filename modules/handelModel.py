from src.programData.thisdir import thisdir
import os

thisdir = thisdir()


def handleModel(AI, customModel=None):
    if AI == "realesrgan-cuda":
        return f"{thisdir}/models/realesrgan-cuda/realesr-animevideov3.pth"
    if AI == "rife-cuda":  # handle rife models in rife file
        return None
    if AI == "custom-models-cuda":
        return f"{thisdir}/models/custom-models-cuda/{customModel}"
