import subprocess
import os

from .Util import currentDirectory

class SceneDetect:
    """
    Class to detect scene changes based on a few parameters
    sceneChangeSsensitivity: This dictates the sensitivity where a scene detect between frames is activated
        - Lower means it is more suseptable to triggering a scene change
        - 
    """
    def __init__(self,
                inputFile:str,
                sceneChangeSensitivity:str,
                sceneChangeMethod:str = "PySceneDetect",
                ):
        self.inputFile = inputFile
        self.sceneChangeSsensitivity = sceneChangeSensitivity
        self.sceneChangeMethod = sceneChangeMethod

    def getPySceneDetectTransitions(self) -> list[int]:
        pass

    def getTransitions(self):
        if self.sceneChangeMethod == "PySceneDetect":
            return self.getPySceneDetectTransitions()