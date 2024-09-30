from scenedetect import AdaptiveDetector, open_video
from tqdm import tqdm
from queue import Queue
from .FFmpeg import FFMpegRender
from threading import Thread
from .Util import bytesTo100x100img
from queue import Queue
import numpy as np

class NPMeanSCDetect:
    """
    takes in an image as np array and calculates the mean, with ability to use it for scene detect and upscale skip
    """

    def __init__(self, sensitivity: int = 2):
        self.i0 = None
        self.i1 = None
        # multiply sensitivity by 10 for more representative results
        self.sensitivity = sensitivity * 10
        
        

    # a simple scene detect based on mean
    def sceneDetect(self, img1):
        if self.i0 is None:
            self.i0 = img1
            self.image0mean = np.mean(self.i0)
            return
        self.i1 = img1
        img1mean = np.mean(self.i1)
        if (
            self.image0mean > img1mean + self.sensitivity
            or self.image0mean < img1mean - self.sensitivity
        ):
            self.image0mean = img1mean
            return True
        self.image0mean = img1mean
        return False

class NPMeanSegmentedSCDetect:
    """
    takes in an image as np array and calculates the mean, with ability to use it for scene detect and upscale skip
    """

    def __init__(self, sensitivity: int = 2, segments: int = 10):
        self.i0 = None
        self.i1 = None
        # multiply sensitivity by 10 for more representative results
        self.sensitivity = sensitivity * 10
        self.segments = segments
    
    def segmentImage(self, img: np.ndarray):
        # split image into segments
        # calculate mean of each segment
        # return list of means
        h, w = img.shape[:2]
        segment_height = h // self.segments
        segment_width = w // self.segments

        means = {}
        for i in range(self.segments):
            for j in range(self.segments):
                segment = img[
                    i * segment_height:(i + 1) * segment_height,
                    j * segment_width:(j + 1) * segment_width
                ]
                means[i] = np.mean(segment)
        
        return means

    # a simple scene detect based on mean
    def sceneDetect(self, img1):
        if self.i0 is None:
            self.i0 = img1
            self.segmentsImg1Mean = self.segmentImage(self.i0)
            return
        self.i1 = img1
        segmentsImg2Mean = self.segmentImage(self.i1)
        for key,value in self.segmentsImg1Mean.items():
            if (
                value > segmentsImg2Mean[key] + self.sensitivity
                or value < segmentsImg2Mean[key] - self.sensitivity
            ):
                self.segmentsImg1Mean = segmentsImg2Mean
                return True
        self.segmentsImg1Mean = segmentsImg2Mean
        return False



# could be an idea
class VSSCDetect:
    def __init__(self, threshold=0.05):
        self.threshold = threshold

    def detect(self, frames):
        if len(frames) < 8:
            raise ValueError("At least 8 frames are required for scene change detection")

        results = []
        for i in range(4, len(frames) - 3):
            prev_diffs = [self.calculate_diff(frames[j], frames[j+1]) for j in range(i-4, i)]
            next_diffs = [self.calculate_diff(frames[j], frames[j+1]) for j in range(i, i+3)]
            
            mean_diff = np.mean(prev_diffs + next_diffs[1:])
            max_diff = np.max(prev_diffs + next_diffs[1:])
            
            std_dev = np.std(prev_diffs + next_diffs[1:])
            
            dynamic_threshold = self.threshold + std_dev
            
            next_diff = self.calculate_diff(frames[i], frames[i+1])
            
            is_scene_change = (next_diff - dynamic_threshold) > max_diff
            results.append(is_scene_change)

        return results

    def calculate_diff(self, frame1, frame2):
        # Assuming frames are numpy arrays
        return np.mean(np.abs(frame1.astype(float) - frame2.astype(float)))
    
class SceneDetect():
    """
    Class to detect scene changes based on a few parameters
    sceneChangeSsensitivity: This dictates the sensitivity where a scene detect between frames is activated
        - Lower means it is more suseptable to triggering a scene change
        -
    """

    def __init__(
        self,
        sceneChangeMethod: str = "mean",
        sceneChangeSensitivity: float = 2.0,
        width: int = 1920,
        height: int = 1080,
    ):
        self.width = width
        self.height = height
        if sceneChangeMethod == "mean":
            self.detector = NPMeanSCDetect(sensitivity=sceneChangeSensitivity)
        elif sceneChangeMethod == "mean_segmented":
            self.detector = NPMeanSegmentedSCDetect(sensitivity=sceneChangeSensitivity, segments=5)

    def detect(self,frame):
        frame = bytesTo100x100img(
                frame, width=self.width, height=self.height
            )
        out = self.detector.sceneDetect(frame)
        return out
    
    


if __name__ == "__main__":
    import sys

    scdetect = SceneDetect(sys.argv[1])
    scdetect.getTransitions()
