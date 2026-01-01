import numpy as np
import os
import cv2
from collections import deque
import sys
from .PySceneDetectUtils import ContentDetector
from ..utils.Frame import Frame
class BaseDetector:
    def __init__(self, threshold: int = 0):
        pass

    def sceneDetect(self, frame: Frame) -> bool:
        return False

class ModelDetector(BaseDetector):
    def __init__(self, threshold: int = 0, model_path: str = None, model_dtype: str = "float32", model_device: str = "cpu"):
        super().__init__(threshold)
    
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
    def sceneDetect(self, frame: Frame):
        if self.i0 is None:
            self.i0 = frame.get_frame_np()
            self.image0mean = np.mean(self.i0)
            return
        self.i1 = frame.get_frame_np()
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
    def sceneDetect(self, img1: Frame):
        img1 = img1.get_frame_np()
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

    def sceneDetect(self, img1: Frame):
        img1 = img1.get_frame_np()
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


class PySceneDetect(BaseDetector):
    def __init__(self, threshold=2):
        self.detector = ContentDetector(
            threshold=threshold * 10, min_scene_len=1
        )  # has to be 1 to stay synced
        self.frameNum = 0

    def sceneDetect(self, frame: Frame):
        frame = cv2.resize(frame.get_np_sdr(), (640, 360))
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

class PyTorchSudoSceneDetect(ModelDetector):
    def __init__(self, threshold=0, model_path="", model_dtype="float32", model_device="cpu", model_backend="pytorch", model_gpu_id=0, **kwargs):
        from ..pytorch.scenechangedetect.PyTorchEfficientNetSC import InferenceSceneChangeDetectEfficientNet
        import torch
        self.torch = torch
        self.model = InferenceSceneChangeDetectEfficientNet(threshold=threshold, model_path=model_path, model_dtype=model_dtype, model_device=model_device, model_backend=model_backend)
        self.i0 = None
    def sceneDetect(self, frame: Frame):
        frame = self.torch.nn.functional.interpolate(frame.get_frame_tensor(), 
                            size=(256, 256), 
                            mode='bilinear', 
                            align_corners=False, 
                            ).squeeze(0)
        if self.i0 is None:
            self.i0 = frame
            self.model.model
            return False
        out = self.model(self.i0, frame)
        self.i0 = frame
        return out

class NCNNSudoSceneDetect(ModelDetector):
    def __init__(self, threshold=0, model_path="", model_dtype="float32", model_device="cpu", **kwargs):
        from ..ncnn.NCNNEfficientNetSC import InferenceSceneChangeDetectEfficientNetNCNN
        self.model = InferenceSceneChangeDetectEfficientNetNCNN(threshold=threshold, model_path=model_path, model_dtype=model_dtype, model_device=model_device)
        self.i0 = None
    def sceneDetect(self, frame: Frame):
        frame = frame.clone().resize_frame(256,256).get_frame_np()
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
        model_path: str = None,
        model_backend: str = "pytorch",
        model_dtype: str = "float32",
        model_device: str = "cpu",
        model_gpu_id: int = 0,
    ):
        self.width = width
        self.height = height
        self.sceneChangeMethod = sceneChangeMethod.lower()
        scmethoddict = {
            "mean": NPMeanSCDetect,
            "mean_diff": NPMeanDiffSCDetect,
            "mean_segmented": NPMeanSegmentedSCDetect,
            "pyscenedetect": PySceneDetect,
            "none": BaseDetector,
        }

        assert self.sceneChangeMethod in scmethoddict or model_path is not None, "Invalid Scene Change Method"
        if self.sceneChangeMethod in scmethoddict:
            self.detector: BaseDetector = scmethoddict[self.sceneChangeMethod](
                threshold=sceneChangeSensitivity
            )
        else:
            assert model_path is not None and os.path.exists(model_path),  "Model path must be provided for model-based scene detection. Please pass --scene_detect_model parameter"
            model = PyTorchSudoSceneDetect if model_backend == "pytorch" or model_backend == "tensorrt" else NCNNSudoSceneDetect
            self.detector: ModelDetector = model(
                threshold=sceneChangeSensitivity,
                model_path=model_path,
                model_dtype=model_dtype,
                model_device=model_device,
                model_backend=model_backend,
                model_gpu_id=model_gpu_id,
            )

    def detect(self, frame: Frame) -> bool:
        return self.detector.sceneDetect(frame)
