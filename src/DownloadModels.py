import os
import requests

from .Util import printAndLog, currentDirectory, warnAndLog, createDirectory, modelsPath
from .QTcustom import DownloadProgressPopup


class DownloadModel:
    """
    Takes in the name of a model and the name of the backend in the GUI, and downloads it from a URL
    model: any valid model used by RVE
    backend: the backend used (pytorch, tensorrt, ncnn)
    """

    def __init__(
        self,
        model: str,
        backend: str,
        modelPath: str = modelsPath(),
    ):
        self.modelPath = modelPath
        # create Model path directory where models will be downloaded
        createDirectory(modelPath)
        modelFile = self.getModelFile(model=model, backend=backend)
        modelPath = os.path.join(modelsPath(), modelFile)
        if not os.path.isfile(modelPath):
            self.downloadModel(modelFile=modelFile, modelPath=modelPath)

    def getModelFile(self, model: str = None, backend: str = None):
        match model:
            case "RIFE 4.20":
                if backend == "ncnn":
                    pass
                else:
                    return "rife4.20.pkl"
            case _:
                raise os.error("Not a valid model!")

    def downloadModel(self, modelFile: str = None, modelPath: str = None):
        url = (
            "https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/"
            + modelFile
        )
        title = "Downloading: " + modelFile
        DownloadProgressPopup(link=url, title=title, downloadLocation=modelPath)


# just some testing code lol
if __name__ == "__main__":
    downloadModels = DownloadModel(
        modelsList=["2x_ModernSpanimationV1.5.pth"],
    )
