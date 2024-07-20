import subprocess
import os
from scenedetect import AdaptiveDetector, open_video
from tqdm import tqdm
import numpy as np
import cv2

from Util import printAndLog

def resizeNPArray():
    pass

class SceneDetect:
    """
    Class to detect scene changes based on a few parameters
    sceneChangeSsensitivity: This dictates the sensitivity where a scene detect between frames is activated
        - Lower means it is more suseptable to triggering a scene change
        - 
    """
    def __init__(self,
                inputFile:str,
                sceneChangeSensitivity:float = 3.,
                sceneChangeMethod:str = "PySceneDetect",
                ):
        self.inputFile = inputFile
        self.sceneChangeSsensitivity = sceneChangeSensitivity
        self.sceneChangeMethod = sceneChangeMethod

    

    def getPySceneDetectTransitions(self) -> list[int]:
        sceneChangeList = []
        adaptiveDetector = AdaptiveDetector(adaptive_threshold=self.sceneChangeSsensitivity)
        openedVideo = open_video(self.inputFile)
        frame_count = openedVideo.duration.frame_num
        for frame_num in tqdm(range(frame_count)):
            frame = openedVideo.read()
            frame = cv2.resize(frame, dsize=(100,100)) # downscaling makes no difference in quality for scene change, bottlenecked by resize speed 
            detectedFrameList = adaptiveDetector.process_frame(frame_num=frame_num,frame_img=frame)
            #if len(detectedFrameList) == 1:
            #    sceneChangeList += detectedFrameList  
            match len(detectedFrameList):
                case 1:
                    sceneChangeList += detectedFrameList  
        print(sceneChangeList)
        
    def getTransitions(self) -> list[int]:
        "Method that returns a list of ints where the scene changes are."
        printAndLog("Detecting Transitions")
        if self.sceneChangeMethod == "PySceneDetect":
            return self.getPySceneDetectTransitions()

if __name__ == "__main__":
    import sys
    scdetect = SceneDetect(sys.argv[1])
    scdetect.getTransitions()