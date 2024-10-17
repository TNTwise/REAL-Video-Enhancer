import numpy as np
import cv2
from collections import deque
from .Util import bytesToImg


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
    takes in an image as np array and calculates the mean, with ability to use it for scene detect
    Args:
        sensitivity: int: sensitivity of the scene detect
        segments: int: number of segments to split the image into
        maxDetections: int: number of detections in a segmented scene to trigger a scene change, default is half the segments
    """

    def __init__(
        self, sensitivity: int = 2, segments: int = 10, maxDetections: int = None
    ):
        self.i0 = None
        self.i1 = None
        if maxDetections is None:
            maxDetections = segments // 2 if segments > 1 else 1
        # multiply sensitivity by 10 for more representative results
        self.sensitivity = sensitivity * 10
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


# could be an idea
class VSSCDetect:
    def __init__(self, threshold=0.05):
        self.threshold = threshold

    def detect(self, frames):
        if len(frames) < 8:
            raise ValueError(
                "At least 8 frames are required for scene change detection"
            )

        results = []
        for i in range(4, len(frames) - 3):
            prev_diffs = [
                self.calculate_diff(frames[j], frames[j + 1]) for j in range(i - 4, i)
            ]
            next_diffs = [
                self.calculate_diff(frames[j], frames[j + 1]) for j in range(i, i + 3)
            ]

            mean_diff = np.mean(prev_diffs + next_diffs[1:])
            max_diff = np.max(prev_diffs + next_diffs[1:])

            std_dev = np.std(prev_diffs + next_diffs[1:])

            dynamic_threshold = self.threshold + std_dev

            next_diff = self.calculate_diff(frames[i], frames[i + 1])

            is_scene_change = (next_diff - dynamic_threshold) > max_diff
            results.append(is_scene_change)

        return results

    def calculate_diff(self, frame1, frame2):
        # Assuming frames are numpy arrays
        return np.mean(np.abs(frame1.astype(float) - frame2.astype(float)))


class NPMeanDiffSCDetect:
    def __init__(self, sensitivity=2):
        self.sensativity = (
            sensitivity * 10
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


class FFMPEGSceneDetect:
    def __init__(self, threshold=0.3, min_scene_length=15, history_size=30):
        self.threshold = threshold
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
        # this is just the argument from the command line, default is mean
        if sceneChangeMethod == "mean":
            self.detector = NPMeanSCDetect(sensitivity=sceneChangeSensitivity)
        elif sceneChangeMethod == "mean_diff":
            self.detector = NPMeanDiffSCDetect(sensitivity=sceneChangeSensitivity)
        elif sceneChangeMethod == "mean_segmented":
            self.detector = NPMeanSegmentedSCDetect(
                sensitivity=sceneChangeSensitivity, segments=4
            )
        elif sceneChangeMethod == "ffmpeg":
            self.detector = FFMPEGSceneDetect(
                threshold=sceneChangeSensitivity / 10,
                min_scene_length=15,
                history_size=30,
            )
        else:
            raise ValueError("Invalid scene change method")

    def detect(self, frame):
        frame = bytesToImg(frame, width=self.width, height=self.height)
        out = self.detector.sceneDetect(frame)
        return out
