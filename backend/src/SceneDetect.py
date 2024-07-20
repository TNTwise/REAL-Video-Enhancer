from scenedetect import AdaptiveDetector, open_video
from tqdm import tqdm
import cv2
from queue import Queue

from .Util import printAndLog


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
                sceneChangeMethod:str = "pyscenedetect",
                ):
        self.inputFile = inputFile
        self.sceneChangeSensitivity = sceneChangeSensitivity
        self.sceneChangeMethod = sceneChangeMethod

    

    def getPySceneDetectTransitions(self) -> Queue:
        sceneChangeStack = Queue()
        adaptiveDetector = AdaptiveDetector(adaptive_threshold=self.sceneChangeSensitivity)
        openedVideo = open_video(self.inputFile)
        frame_count = openedVideo.duration.frame_num
        for frame_num in tqdm(range(frame_count - 1)):
            frame = openedVideo.read()
            frame = cv2.resize(frame, dsize=(100,100)) # downscaling makes no difference in quality for scene change, bottlenecked by resize speed 
            detectedFrameList = adaptiveDetector.process_frame(frame_num=frame_num,frame_img=frame)
            #if len(detectedFrameList) == 1:
            #    sceneChangeList += detectedFrameList  
            match len(detectedFrameList):
                case 1:
                    sceneChangeStack.put(frame_num)
        return sceneChangeStack

    def getTransitions(self) -> Queue:
        "Method that returns a list of ints where the scene changes are."
        printAndLog("Detecting Transitions")
        if self.sceneChangeMethod == "pyscenedetect":
            return self.getPySceneDetectTransitions()

if __name__ == "__main__":
    import sys
    scdetect = SceneDetect(sys.argv[1])
    scdetect.getTransitions()