import numpy as np


class NPMeanSequential:
    """
    takes in an image as np array and calculates the mean, with ability to use it for scene detect and upscale skip
    """

    def __init__(self, sensitivity: int = 2):
        self.i0 = None
        self.i1 = None
        # multiply sensitivity by 10 for more representative results
        self.sensitivity = sensitivity * 10

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

    def isEqualImages(self, img1: np.ndarray):
        if self.i0 is None:
            self.i0 = img1
            return
        self.i1: np.ndarray = img1
        if np.array_equal(self.i0, self.i1):
            self.i0: np.ndarray = self.i1
            print("Skipped upscaling frame")
            return True
        self.i0 = self.i1
        return False

# could be an idea
class SCDetect:
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