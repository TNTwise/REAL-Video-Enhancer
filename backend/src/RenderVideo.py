from threading import Thread
from queue import Queue

from .FFmpeg import FFMpegRender
from .SceneDetect import SceneDetect
from .Util import printAndLog

# try/except imports
try:
    from .UpscaleNCNN import UpscaleNCNN, getNCNNScale
    from .InterpolateNCNN import InterpolateRIFENCNN
except ImportError:
    print("WARN: unable to import ncnn.")

try:
    from .InterpolateTorch import InterpolateRifeTorch
    from .UpscaleTorch import UpscalePytorch
except ImportError:
    print("WARN: unable to import pytorch.")


class Render(FFMpegRender):
    """
    Subclass of FFmpegRender
    FFMpegRender options:
    inputFile: str, The path to the input file.
    outputFile: str, The path to the output file.
    interpolateTimes: int, this sets the multiplier for the framerate when interpolating, when only upscaling this will be set to 1.
    encoder: str, The exact name of the encoder ffmpeg will use (default=libx264)
    pixelFormat: str, The pixel format ffmpeg will use, (default=yuv420p)

    RenderOptions:
    interpolationMethod
    upscaleModel
    backend (pytorch,ncnn,tensorrt)
    device (cpu,cuda)
    precision (float16,float32)
    """

    def __init__(
        self,
        inputFile: str,
        outputFile: str,
        # backend settings
        backend="pytorch",
        device="cuda",
        precision="float16",
        # model settings
        upscaleModel=None,
        interpolateModel=None,
        interpolateFactor: int = 1,
        interpolateArch: str = "rife413",
        # ffmpeg settings
        encoder: str = "libx264",
        pixelFormat: str = "yuv420p",
        benchmark: bool = False,
        overwrite: bool = False,
        crf: str = "18",
        # misc
        sceneDetectMethod: str = "pyscenedetect",
        sceneDetectSensitivity: float = 3.0,
    ):
        self.inputFile = inputFile
        self.backend = backend
        self.upscaleModel = upscaleModel
        self.interpolateModel = interpolateModel
        self.interpolateArch = interpolateArch
        self.device = device
        self.precision = precision
        self.upscaleTimes = 1  # if no upscaling, it will default to 1
        self.interpolateFactor = interpolateFactor
        self.setupRender = self.returnFrame  # set it to not convert the bytes to array by default, and just pass chunk through
        self.frame0 = None
        self.sceneDetectMethod = sceneDetectMethod
        self.sceneDetectSensitivty = sceneDetectSensitivity

        self.getVideoProperties(inputFile)
        printAndLog("Using backend: " + self.backend)
        if upscaleModel:
            self.setupUpscale()
            self.renderThread = Thread(target=self.renderUpscale)
            printAndLog("Using Upscaling Model: " + self.interpolateModel)
        if interpolateModel:
            self.setupInterpolate()
            self.renderThread = Thread(target=self.renderInterpolate)
            printAndLog("Using Interpolation Model: " + self.interpolateModel)

        super().__init__(
            inputFile=inputFile,
            outputFile=outputFile,
            interpolateFactor=interpolateFactor,
            upscaleTimes=self.upscaleTimes,
            encoder=encoder,
            pixelFormat=pixelFormat,
            benchmark=benchmark,
            overwrite=overwrite,
            frameSetupFunction=self.setupRender,
            crf=crf,
        )
        self.ffmpegReadThread = Thread(target=self.readinVideoFrames)
        self.ffmpegWriteThread = Thread(target=self.writeOutVideoFrames)

        self.ffmpegReadThread.start()
        self.ffmpegWriteThread.start()
        self.renderThread.start()

    def renderUpscale(self):
        """
        self.setupRender, method that is mapped to the bytesToFrame in each respective backend
        self.upscale, method that takes in a chunk, and outputs an array that can be sent to ffmpeg
        """
        printAndLog("Starting Upscale")
        for i in range(self.totalFrames - 1):
            frame = self.readQueue.get()
            frame = self.upscale(frame)
            self.writeQueue.put(frame)
        self.writeQueue.put(None)
        printAndLog("Finished Upscale")

    def renderInterpolate(self):
        """
        self.setupRender, method that is mapped to the bytesToFrame in each respective backend
        self.interpoate, method that takes in a chunk, and outputs an array that can be sent to ffmpeg
        """
        printAndLog("Starting Interpolation")
        self.transitionFrame = -1
        self.frame0 = self.readQueue.get()

        for frameNum in range(self.totalFrames - 1):
            frame1 = self.readQueue.get()
            if frame1 is None:
                break
            if self.transitionFrame is None or frameNum != self.transitionFrame:
                for n in range(self.interpolateFactor):
                    frame = self.interpolate(
                        self.frame0, frame1, 1 / (self.interpolateFactor - n)
                    )
                    self.writeQueue.put(frame)
            else:
                # undo the setup done in ffmpeg thread
                sc_detected_frame_np = self.undoSetup(
                    self.frame0
                )
                for n in range(self.interpolateFactor):
                    self.writeQueue.put(sc_detected_frame_np)
                try:  # get_nowait sends an error out of the queue is empty, I would like a better solution than this though
                    self.transitionFrame = self.transitionQueue.get_nowait()
                except:
                    self.transitionFrame = None
            self.frame0 = frame1

        self.writeQueue.put(None)
        printAndLog("Finished Interpolation")

    def setupUpscale(self):
        """
        This is called to setup an upscaling model if it exists.
        Maps the self.upscaleTimes to the actual scale of the model
        Maps the self.setupRender function that can setup frames to be rendered
        Maps the self.upscale the upscale function in the respective backend.
        For interpolation:
        Mapss the self.undoSetup to the tensor_to_frame function, which undoes the prep done in the FFMpeg thread. Used for SCDetect
        """
        printAndLog("Setting up Upscale")
        if self.backend == "pytorch" or self.backend == "tensorrt":
            upscalePytorch = UpscalePytorch(
                self.upscaleModel,
                device=self.device,
                precision=self.precision,
                width=self.width,
                height=self.height,
                backend=self.backend,
            )
            self.upscaleTimes = upscalePytorch.getScale()
            self.setupRender = upscalePytorch.bytesToFrame
            self.upscale = upscalePytorch.renderToNPArray

        if self.backend == "ncnn":
            self.upscaleTimes = getNCNNScale(modelPath=self.upscaleModel)
            upscaleNCNN = UpscaleNCNN(
                modelPath=self.upscaleModel,
                num_threads=1,
                scale=self.upscaleTimes,
                gpuid=0,  # might have this be a setting
                width=self.width,
                height=self.height,
            )
            self.setupRender = self.returnFrame
            self.upscale = upscaleNCNN.Upscale

    def setupInterpolate(self):
        printAndLog("Setting up Interpolation")

        if self.sceneDetectMethod != "none":
            printAndLog("Detecting Transitions")
            scdetect = SceneDetect(
                inputFile=self.inputFile,
                sceneChangeSensitivity=self.sceneDetectSensitivty,
                sceneChangeMethod=self.sceneDetectMethod,
            )
            self.transitionQueue = scdetect.getTransitions()
        else:
            self.transitionQueue = None
        if self.backend == "ncnn":
            interpolateRifeNCNN = InterpolateRIFENCNN(
                interpolateModelPath=self.interpolateModel,
                width=self.width,
                height=self.height,
            )
            self.setupRender = interpolateRifeNCNN.bytesToByteArray
            self.undoSetup = self.returnFrame
            self.interpolate = interpolateRifeNCNN.process
        if self.backend == "pytorch" or self.backend == "tensorrt":
            interpolateRifePytorch = InterpolateRifeTorch(
                interpolateModelPath=self.interpolateModel,
                interpolateArch=self.interpolateArch,
                width=self.width,
                height=self.height,
                device=self.device,
                dtype=self.precision,
                backend=self.backend,
            )
            self.setupRender = interpolateRifePytorch.frame_to_tensor
            self.undoSetup = interpolateRifePytorch.tensor_to_frame
            self.interpolate = interpolateRifePytorch.process
