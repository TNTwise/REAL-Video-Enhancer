from scenedetect import AdaptiveDetector, open_video
from tqdm import tqdm
from queue import Queue
from .FFmpeg import FFMpegRender
from threading import Thread
from .Util import bytesTo100x100img
from .NPMean import NPMeanSequential
from queue import Queue


class SceneDetect(FFMpegRender):
    """
    Class to detect scene changes based on a few parameters
    sceneChangeSsensitivity: This dictates the sensitivity where a scene detect between frames is activated
        - Lower means it is more suseptable to triggering a scene change
        -
    """

    def __init__(
        self,
        inputFile: str,
        sceneChangeSensitivity: float = 3.0,
        sceneChangeMethod: str = "pyscenedetect",
    ):
        self.getVideoProperties(inputFile)
        super().__init__(
            inputFile=inputFile,
            outputFile=None,
            interpolateFactor=1,
            upscaleTimes=1,
            benchmark=True,
            overwrite=True,
            sharedMemoryID=None,
            channels=3,
        )
        self.getVideoProperties(inputFile)
        self.inputFile = inputFile
        self.sceneChangeSensitivity = sceneChangeSensitivity
        self.sceneChangeMethod = sceneChangeMethod
        self.sceneChangeQueue = Queue()
        self.adaptiveDetector = AdaptiveDetector(
            adaptive_threshold=self.sceneChangeSensitivity
        )
        self.meanDetector = NPMeanSequential(sensitivity=self.sceneChangeSensitivity)

        self.readThread = Thread(target=self.readinVideoFrames)

        self.frameNum = 0

        # add frame chunk size to ffmpegrender



    def getPySceneDetectTransitions(self) -> Queue:
        self.readThread.start()
        

        for frame_num in tqdm(range(self.totalInputFrames - 1)):
            frame = bytesTo100x100img(
                self.readQueue.get(), width=self.width, height=self.height
            )

            detectedFrameList = self.adaptiveDetector.process_frame(
                frame_num=frame_num, frame_img=frame
            )
            # if len(detectedFrameList) == 1:
            #    sceneChangeList += detectedFrameList
            match len(detectedFrameList):
                case 1:
                    self.sceneChangeQueue.put(detectedFrameList[0])
        return self.sceneChangeQueue

    def getMeanTransitions(self):
        self.readThread.start()
        
        
        for frame_num in tqdm(range(self.totalInputFrames - 1)):
            frame = bytesTo100x100img(
                self.readQueue.get(), width=self.width, height=self.height
            )
            if self.meanDetector.sceneDetect(frame):
                self.sceneChangeQueue.put(frame_num )
        return self.sceneChangeQueue
    
    def processMeanTransition(self,frame):
        frame = bytesTo100x100img(
                frame, width=self.width, height=self.height
            )
        out = self.meanDetector.sceneDetect(frame)
        return out
    
    def processPySceneDetectTransition(self,frame):
        frame = bytesTo100x100img(
                frame, width=self.width, height=self.height
            )
        detectedFrameList = self.adaptiveDetector.process_frame(
                frame_num=0, frame_img=frame
            )
        self.frameNum += 1
        if len(detectedFrameList) > 0:
            exit()

    def getTransitions(self) -> Queue:
        "Method that returns a list of ints where the scene changes are."

        if self.sceneChangeMethod == "pyscenedetect":
            return self.getPySceneDetectTransitions()
        if self.sceneChangeMethod == "mean":
            return self.getMeanTransitions()


if __name__ == "__main__":
    import sys

    scdetect = SceneDetect(sys.argv[1])
    scdetect.getTransitions()
