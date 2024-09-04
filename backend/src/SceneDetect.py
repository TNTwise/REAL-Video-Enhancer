from scenedetect import AdaptiveDetector, open_video
from tqdm import tqdm
import cv2
from queue import Queue
import numpy as np
from .FFmpeg import FFMpegRender
from threading import Thread
from .Util import printAndLog
from queue import Queue
class SCDetectNP:
    def __init__(self):
        self.i0 = None
        self.i1 = None
    def forward(self, img1):
        if self.i0 is None:
            self.i0 = img1
            self.image0mean = np.mean(self.i0)
            return
        self.i1 = img1
        img1mean = np.mean(self.i1)
        if self.image0mean > img1mean + 20 or self.image0mean < img1mean - 20:
            self.image0mean = img1mean
            return True
        self.image0mean = img1mean
        return False
        
        

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
            channels=3
            )
        self.getVideoProperties(inputFile)
        self.inputFile = inputFile
        self.sceneChangeSensitivity = sceneChangeSensitivity
        self.sceneChangeMethod = sceneChangeMethod
        
        
        self.readThread = Thread(target=self.readinVideoFrames)
        
        # add frame chunk size to ffmpegrender
    
    def getFrameTo100x100img(self, image: bytes) -> np.ndarray:
        frame = np.frombuffer(image,dtype=np.uint8).reshape(self.height, self.width, 3)
        frame = cv2.resize(
                frame, dsize=(100, 100)
            )
        return frame

    def copy_queue(self,original_queue):
        items = []
        
        # Extract items to a list
        while not original_queue.empty():
            items.append(original_queue.get())
        
        new_queue = Queue()
        for item in items:
            new_queue.put(item)
            original_queue.put(item) 
    
        return new_queue

    def printQueue(self,queue: Queue):
        queue = self.copy_queue(queue)
        t = []
        while not queue.empty():
            item = queue.get()
            t.append(item)
        print(t)
        print(len(t))

    def getPySceneDetectTransitions(self) -> Queue:
        self.readThread.start()
        sceneChangeQueue = Queue()
        adaptiveDetector = AdaptiveDetector(
            adaptive_threshold=self.sceneChangeSensitivity
        )
        
        for frame_num in tqdm(range(self.totalInputFrames - 1)):
            
            frame = self.getFrameTo100x100img(self.readQueue.get())
            
            detectedFrameList = adaptiveDetector.process_frame(
                frame_num=frame_num, frame_img=frame
            )
            # if len(detectedFrameList) == 1:
            #    sceneChangeList += detectedFrameList
            match len(detectedFrameList):
                case 1:
                    sceneChangeQueue.put(detectedFrameList[0] - 1)
        return sceneChangeQueue
        
        
    
    def getMeanTransitions(self):
        self.readThread.start()
        sceneChangeQueue = Queue()
        detector = SCDetectNP()
        for frame_num in tqdm(range(self.totalInputFrames - 1)):
            frame = self.getFrameTo100x100img(self.readQueue.get())
            if detector.forward(frame):
                sceneChangeQueue.put(frame_num-1)
        return sceneChangeQueue

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
