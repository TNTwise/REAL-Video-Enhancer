import os

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
        backend: str,
        modelPath: str = modelsPath(),
    ):
        self.modelPath = modelPath
        # get initial extension
        self.modelFileExtension = modelFile.split(".")[-1]
        # create Model path directory where models will be downloaded
        createDirectory(modelPath)
        modelPath = os.path.join(modelsPath(), modelFile)
        modelFile = self.getModelFileToDownload(modelFile=modelFile)
        # override, necessary if it is tar.gz
        self.downloadModelFileExtension = modelFile.split(".")[-1]
        if not os.path.isfile(modelPath):
            self.downloadModel(modelFile=modelFile, modelPath=modelPath)

    def getModelFileToDownload(self, modelFile: str = None):
        
        if self.modelFileExtension == ".pth" or self.modelFileExtension == ".pkl":
            return modelFile
        return modelFile + ".tar.gz"
        

    def downloadModel(self, modelFile: str = None, modelPath: str = None):
        url = (
            "https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/"
            + modelFile
        )
        title = "Downloading: " + modelFile
        DownloadProgressPopup(link=url, title=title, downloadLocation=modelPath)
        if self.downloadModelFileExtension == ".tar.gz":
            extractTarGZ(modelFile)

# just some testing code lol
