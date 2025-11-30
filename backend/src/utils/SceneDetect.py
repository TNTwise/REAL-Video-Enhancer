import numpy as np
import cv2
from collections import deque
import sys
from .Util import bytesToImg
from .PySceneDetectUtils import ContentDetector


class BaseDetector:
    def __init__(self, threshold: int = 0):
        pass

    def sceneDetect(self, frame):
        return False


class NPMeanSCDetect(BaseDetector):
    """
    takes in an image as np array and calculates the mean, with ability to use it for scene detect and upscale skip
    """

    def __init__(self, threshold: int = 2):
        self.i0 = None
        self.i1 = None
        # multiply sensitivity by 10 for more representative results
        self.sensitivity = threshold * 10

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


class NPMeanSegmentedSCDetect(BaseDetector):
    """
    takes in an image as np array and calculates the mean, with ability to use it for scene detect
    Args:
        sensitivity: int: sensitivity of the scene detect
        segments: int: number of segments to split the image into
        maxDetections: int: number of detections in a segmented scene to trigger a scene change, default is half the segments
    """

    def __init__(
        self, threshold: int = 2, segments: int = 10, maxDetections: int = None
    ):
        self.i0 = None
        self.i1 = None
        if maxDetections is None:
            maxDetections = segments // 2 if segments > 1 else 1
        # multiply sensitivity by 10 for more representative results
        self.sensitivity = threshold * 10
        self.segments = segments
        self.maxDetections = maxDetections

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
                    i * segment_height : (i + 1) * segment_height,
                    j * segment_width : (j + 1) * segment_width,
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
        detections = 0
        for key, value in self.segmentsImg1Mean.items():
            if (
                value > segmentsImg2Mean[key] + self.sensitivity
                or value < segmentsImg2Mean[key] - self.sensitivity
            ):
                self.segmentsImg1Mean = segmentsImg2Mean
                detections += 1
                if detections >= self.maxDetections:
                    return True
        self.segmentsImg1Mean = segmentsImg2Mean
        return False


class NPMeanDiffSCDetect(BaseDetector):
    def __init__(self, threshold=2):
        self.sensativity = (
            threshold * 10
        )  # multiply by 10 for more representative results
        self.i0 = None
        self.i1 = None

    def sceneDetect(self, img1):
        if self.i0 is None:
            self.i0 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            return

        self.i1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(self.i1, self.i0)

        mean_diff = np.mean(frame_diff)
        if mean_diff > self.sensativity:
            self.i0 = self.i1
            return True
        self.i0 = self.i1
        return False


class FFMPEGSceneDetect(BaseDetector):
    def __init__(self, threshold=0.3, min_scene_length=1, history_size=30):
        self.threshold = threshold / 10
        self.min_scene_length = min_scene_length
        self.history_size = history_size
        self.frame_diffs = deque(maxlen=history_size)
        self.hist_diffs = deque(maxlen=history_size)
        self.prev_frame = None
        self.frames_since_last_scene = 0

    def compute_frame_difference(self, frame1, frame2):
        # Convert to YUV color space
        yuv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2YUV)
        yuv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2YUV)

        # Compute difference in Y (luminance) channel
        diff_y = cv2.absdiff(yuv1[:, :, 0], yuv2[:, :, 0])

        # Compute histogram difference
        hist1 = cv2.calcHist([yuv1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([yuv2], [0], None, [256], [0, 256])
        hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

        return np.mean(diff_y), hist_diff

    def sceneDetect(self, frame):
        if self.prev_frame is None:
            self.prev_frame = frame
            return False

        diff_y, hist_diff = self.compute_frame_difference(self.prev_frame, frame)
        self.frame_diffs.append(diff_y)
        self.hist_diffs.append(hist_diff)

        self.prev_frame = frame
        self.frames_since_last_scene += 1

        if len(self.frame_diffs) < self.history_size:
            return False

        # Combine frame and histogram differences
        combined_diff = np.array(self.frame_diffs) * np.array(self.hist_diffs)

        # Normalize the differences
        normalized_diff = (combined_diff - np.min(combined_diff)) / (
            np.max(combined_diff) - np.min(combined_diff)
        )

        # Apply moving average filter
        window_size = 5
        smoothed_diff = np.convolve(
            normalized_diff, np.ones(window_size) / window_size, mode="valid"
        )

        # Check if the latest smoothed difference exceeds the threshold
        if (
            smoothed_diff[-1] > self.threshold
            and self.frames_since_last_scene >= self.min_scene_length
        ):
            self.frames_since_last_scene = 0
            return True

        return False


class PySceneDetect(BaseDetector):
    def __init__(self, threshold=2, min_scene_length=30):
        self.detector = ContentDetector(
            threshold=threshold * 10, min_scene_len=1
        )  # has to be 1 to stay synced
        self.frameNum = 0

    def sceneDetect(self, frame: np.ndarray):
        frame = cv2.resize(frame, (640, 360))
        frameList = self.detector.process_frame(self.frameNum, frame)
        self.frameNum += 1
        if len(frameList) > 0:
            if self.frameNum != frameList[0] + 1:
                print(
                    f"Transition Mismatch {self.frameNum} is not equal to {frameList[0] + 1}, skipping",
                    file=sys.stderr,
                )
                return False

        return len(frameList) > 0

class PyTorchSudoSceneDetect(BaseDetector):
    def __init__(self, **kwargs):
        from ..pytorch.scenechangedetect.PyTorchEfficientNetSC import InferenceSceneChangeDetectEfficientNet
        import torch
        from ..pytorch.TorchUtils import TorchUtils
        self.torch = torch
        self.model = InferenceSceneChangeDetectEfficientNet()
        self.i0 = None
    def sceneDetect(self, frame):
        frame = self.torch.from_numpy(frame).to(dtype=self.torch.float16, device='cuda').permute(2, 0, 1).div(255.)
        frame = self.torch.nn.functional.interpolate(frame.unsqueeze(0), 
                            size=(256, 256), 
                            mode='bilinear', 
                            align_corners=False, 
                            ).squeeze(0)
        if self.i0 is None:
            self.i0 = frame
            return False
        out = self.model(self.i0, frame)
        self.i0 = frame
        return out

class RVESceneDetect(BaseDetector):
    def __init__(self, **kwargs):
        self.pass2 = PyTorchSudoSceneDetect()
        self.pass1 = PySceneDetect()
    def sceneDetect(self, frame):
        return self.pass1.sceneDetect(frame) or self.pass2.sceneDetect(frame)
        

class SceneDetect:
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
        self.sceneChangeMethod = sceneChangeMethod.lower()
        scmethoddict = {
            "mean": NPMeanSCDetect,
            "mean_diff": NPMeanDiffSCDetect,
            "mean_segmented": NPMeanSegmentedSCDetect,
            "ffmpeg": FFMPEGSceneDetect,
            "pyscenedetect": PySceneDetect,
            "sudo_pytorch": PyTorchSudoSceneDetect,
            # "rve": RVESceneDetect,
            "none": BaseDetector,
        }

        assert self.sceneChangeMethod in scmethoddict, "Invalid Scene Change Method"
        self.detector: BaseDetector = scmethoddict[self.sceneChangeMethod](
            threshold=sceneChangeSensitivity
        )

    def detect(self, frame):
        if self.sceneChangeMethod != "none":
            frame = bytesToImg(frame, width=self.width, height=self.height)
        
        out = self.detector.sceneDetect(frame)
        return out
