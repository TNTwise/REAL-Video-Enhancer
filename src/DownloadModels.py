import os
import requests

from Util import printAndLog, currentDirectory, warnAndLog, createDirectory


class DownloadModels:
    def __init__(
        self,
        modelsList: list[str],
        modelPath: str = os.path.join(currentDirectory(), "models"),
    ):
        self.modelsList = modelsList
        self.modelPath = modelPath
        # create Model path directory where models will be downloaded
        createDirectory(modelPath)
        # validate that all the models in modelList are correct
        self.validateModelsList()
        self.downloadModels()

    def validateModelsList(self):
        validInterpolateModels = (
            "rife4.6.pkl",
            "rife4.15.pkl",
            "rife4.17.pkl",
        )
        validUpscaleModels = ("2x_ModernSpanimationV1.5.pth",)
        validTotalModels = validInterpolateModels + validUpscaleModels
        for model in self.modelsList:
            if model not in validTotalModels:
                warnAndLog("Not a valid model")

    def downloadModels(self):
        """
        recursivly goes throgh self.modelsList, and calls downloadModel on each one.
        """
        for model in self.modelsList:
            printAndLog("Downloading model: " + model)
            response = requests.get(
                "https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/"
                + model,
                stream=True,
            )

            with open(os.path.join(self.modelPath, model), "wb") as f:
                for chunk in response.iter_content(chunk_size=128):
                    f.write(chunk)


# just some testing code lol
if __name__ == "__main__":
    downloadModels = DownloadModels(
        modelsList=["2x_ModernSpanimationV1.5.pth"],
    )
