import os

from backend.src.Util import printAndLog

from .Util import createDirectory, modelsPath, extractTarGZ
from .QTcustom import DownloadProgressPopup


class DownloadModel:
    """
    Takes in the name of a model and the name of the backend in the GUI, and downloads it from a URL
    model: any valid model used by RVE
    backend: the backend used (pytorch, tensorrt, ncnn)
    """

    def __init__(
        self,
        modelFile: str,
        downloadModelFile:str,
        backend: str,
        modelPath: str = modelsPath(),
    ):
        self.modelPath = modelPath
        self.downloadModelFile = downloadModelFile
        self.downloadModelPath = os.path.join(modelPath,downloadModelFile)
        createDirectory(modelPath)
        


        if not os.path.isfile(modelFile):
            self.downloadModel(modelFile=downloadModelFile, downloadModelPath=self.downloadModelPath)

        

    def downloadModel(self, modelFile: str = None, downloadModelPath: str = None):
        url = (
            "https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/"
            + modelFile
        )
        title = "Downloading: " + modelFile
        DownloadProgressPopup(link=url, title=title, downloadLocation=downloadModelPath)
        if "tar.gz" in self.downloadModelFile:
            print("Extracting File")
            extractTarGZ(self.downloadModelPath)
        


# just some testing code lol
