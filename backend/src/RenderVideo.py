from threading import Thread
from queue import Queue
from multiprocessing import shared_memory

from .FFmpeg import FFMpegRender
from .SceneDetect import SceneDetect
from .Util import printAndLog, log

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

    NOTE:
    Everything in here has to happen in a specific order:
    Get the video properties (res,fps,etc)
    set up upscaling/interpolation, this gets the scale for upscaling if upscaling is the current task
    assign framechunksize to a value, as this is needed to catch bytes and set up shared memory
    set up shared memory
    """

    def __init__(
        self,
        inputFile: str,
        outputFile: str,
        # backend settings
        backend="pytorch",
        device="default",
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
        sharedMemoryID: str = None,
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
        self.sharedMemoryID = sharedMemoryID
        # get video properties early
        self.getVideoProperties(inputFile)

        printAndLog("Using backend: " + self.backend)
        if upscaleModel:
            self.setupUpscale()
            self.renderThread = Thread(target=self.renderUpscale)
            printAndLog("Using Upscaling Model: " + self.upscaleModel)
        if interpolateModel:
            self.setupInterpolate()
            self.renderThread = Thread(target=self.renderInterpolate)
            printAndLog("Using Interpolation Model: " + self.interpolateModel)

        self.inputFrameChunkSize = self.width * self.height * 3
        self.outputFrameChunkSize = (
            self.width * self.upscaleTimes * self.height * self.upscaleTimes * 3
        )
        self.shm = shared_memory.SharedMemory(
            name=self.sharedMemoryID, create=True, size=self.outputFrameChunkSize
        )
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
            sharedMemoryID=sharedMemoryID,
            shm=self.shm,
            inputFrameChunkSize=self.inputFrameChunkSize,
            outputFrameChunkSize=self.outputFrameChunkSize,
        )
        if sharedMemoryID is not None:
            self.sharedMemoryThread = Thread(target=self.writeOutToSharedMemory)
            self.sharedMemoryThread.start()
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
        log("Starting Upscale")
        for i in range(self.totalInputFrames - 1):
            frame = self.readQueue.get()
            frame = self.upscale(self.frameSetupFunction(frame))
            self.writeQueue.put(frame)
        self.writeQueue.put(None)
        log("Finished Upscale")

    def renderInterpolate(self):
        """
        self.setupRender, method that is mapped to the bytesToFrame in each respective backend
        self.interpoate, method that takes in a chunk, and outputs an array that can be sent to ffmpeg
        self.frame0 is always setup,
        frame1 is in bytes, and is only converted if need be
        """
        log("Starting Interpolation")
        self.transitionFrame = self.transitionQueue.get()
        self.frame0 = self.frameSetupFunction(self.readQueue.get())

        for frameNum in range(self.totalInputFrames - 1):
            frame1 = self.readQueue.get()
            if frame1 is None:
                break
            if frameNum != self.transitionFrame:
                for n in range(self.interpolateFactor):
                    timestep = 1 / (self.interpolateFactor - n)
                    if timestep == 1:
                        self.writeQueue.put(frame1)
                        continue

                    frame = self.interpolate(
                        self.frame0, self.frameSetupFunction(frame1), timestep
                    )
                    self.writeQueue.put(frame)
            else:
                # undo the setup done in ffmpeg thread

                for n in range(self.interpolateFactor):
                    self.writeQueue.put(frame1)
                try:  # get_nowait sends an error out of the queue is empty, I would like a better solution than this though
                    self.transitionFrame = self.transitionQueue.get_nowait()
                except:
                    self.transitionFrame = None
            self.frame0 = self.frameSetupFunction(frame1)

        self.writeQueue.put(None)
        log("Finished Interpolation")

    def setupUpscale(self):
        """
        This is called to setup an upscaling model if it exists.
        Maps the self.upscaleTimes to the actual scale of the model
        Maps the self.setupRender function that can setup frames to be rendered
        Maps the self.upscale the upscale function in the respective backend.
        For interpolation:
        Mapss the self.undoSetup to the tensor_to_frame function, which undoes the prep done in the FFMpeg thread. Used for SCDetect
        """
        log("Setting up Upscale")
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
        log("Setting up Interpolation")

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
