import subprocess
import numpy as np
from queue import Queue

import time
from src.programData.thisdir import thisdir

thisdir = thisdir()
import cv2
from src.programData.settings import *
import src.programData.return_data as return_data
from time import sleep

try:
    from .rife.rife import *
    from .UpscaleImage import UpscaleCUDA

except:
    pass
from .UpscaleImageNCNN import UpscaleNCNN, UpscaleCuganNCNN

try:
    from src.torch.gmfss.gmfss_fortuna_union import GMFSS
except:
    pass
try:
    import tensorrt
    from src.torch.rife.tensorRT import RifeTensorRT
    from src.torch.UpscaleImageTensorRT import UpscaleTensorRT
    from polygraphy.backend.trt import (
        TrtRunner,
        engine_from_network,
        network_from_onnx_path,
        CreateConfig,
        Profile,
        EngineFromBytes,
        SaveEngine,
    )
    from polygraphy.backend.common import BytesFromPath

except Exception as e:
    log("Cant import UpscaleTRT!" + str(e))

from src.torch.rife.NCNN import RifeNCNN


# read
# Calculate eta by time remaining divided by speed
# add scenedetect by if frame_num in transitions in proc_frames
# def
class Render:
    def __init__(
        self,
        main,
        input_file,
        output_file,
        interpolationIncrease=1,
        resIncrease=1,
        benchmark=False,
    ):
        self.main = main

        self.readBuffer = Queue(maxsize=50)
        self.writeBuffer = Queue(maxsize=50)
        self.interpolation_factor = round(interpolationIncrease)
        self.prevFrame = None
        self.input_file = input_file
        self.output_file = output_file

        cap = cv2.VideoCapture(input_file)
        self.initialFPS = cap.get(cv2.CAP_PROP_FPS)
        self.originalWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.outputWidth = self.originalWidth * resIncrease
        self.originalHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.outputHeight = self.originalHeight * resIncrease
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.finalFPS = self.initialFPS * self.interpolation_factor
        self.benchmark = benchmark
        self.settings = Settings()

        self.readProcess = None
        self.writeProcess = None

    def extractFramesToBytes(self):
        command = [
            f"{thisdir}/bin/ffmpeg",
            "-i",
            f"{self.input_file}",
            "-f",
            "image2pipe",
            "-pix_fmt",
            "rgb24",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{self.originalWidth}x{self.originalHeight}",
            "-",
        ]

        self.readProcess = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.frame_size = self.originalWidth * self.originalHeight * 3

    def readThread(self):
        while True:
            chunk = self.readProcess.stdout.read(self.frame_size)
            if len(chunk) < self.frame_size:
                self.readProcess.stdout.close()
                self.readProcess.terminate()
                self.readingDone = True
                self.readBuffer.put(None)
                log("done with read")
                break

            self.readBuffer.put(chunk)

    def finish_render(self):
        self.writeBuffer.put(None)

    def returnLatestFrame(self) -> np.ndarray:
        if self.prevFrame:
            return self.prevFrame

    def returnFrameCount(self) -> int:
        try:
            return self.frame
        except:
            log("No frame to return!")

    def log(self):
        while not self.main.CudaRenderFinished:
            try:
                for line in iter(self.writeProcess.stderr.readline, b""):
                    if not self.main.CudaRenderFinished:
                        log(line)

                    else:
                        break

            except Exception as e:
                pass

        log("Done with logging")

    # save

    def FFmpegOut(self):
        log("saving")
        crf = return_data.returnCRFFactor(
            self.settings.videoQuality, self.settings.Encoder
        )

        if not self.benchmark:
            command = [
                f"{thisdir}/bin/ffmpeg",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{self.outputWidth}x{self.outputHeight}",
                "-r",
                f"{self.finalFPS}",
                "-i",
                "-",
            ]
            encoder_list = return_data.returnCodec(self.settings.Encoder).split(" ")
            if os.path.isfile(
                f"{self.settings.RenderDir}/{self.main.videoName}_temp/audio.m4a"
            ):
                command += [
                    "-i",
                    f"{self.settings.RenderDir}/{self.main.videoName}_temp/audio.m4a",
                ]
            command += ["-c:v"]
            command += encoder_list
            command += [
                f"-crf",
                f'{crf.replace("-crf","")}',
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "copy",
                f"{self.output_file}",
            ]
        else:
            command = [
                f"{thisdir}/bin/ffmpeg",
                "-y",
                "-v",
                "warning",
                "-stats",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{self.outputWidth}x{self.outputHeight}",
                "-pix_fmt",
                f"yuv420p",
                "-r",
                str(self.finalFPS),
                "-i",
                "-",
                "-benchmark",
                "-f",
                "null",
                "-",
            ]
        log(command)
        self.writeProcess = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            universal_newlines=True,
        )

        while True:
            frame = self.writeBuffer.get()
            if frame is None:
                self.writeProcess.stdin.close()
                self.writeProcess.wait()
                log("done with save")
                self.main.output_file = self.output_file
                self.main.CudaRenderFinished = True
                torch.cuda.empty_cache()
                break

            # frame = np.ascontiguousarray(frame)
            self.main.imageDisplay = frame
            self.writeProcess.stdin.buffer.write(frame)


class Interpolation(Render):
    def __init__(
        self,
        main,
        method,
        input_file,
        output_file,
        model,
        times,
        ensemble,
        half,
        benchmark,
        ncnn_gpu=0,
        threads=2,
        guiLog=None,
    ):
        super(Interpolation, self).__init__(
            main,
            input_file,
            output_file,
            interpolationIncrease=times,
            resIncrease=1,
            benchmark=benchmark,
        )
        self.guiLog = guiLog
        self.method = method
        self.model = model
        self.ensemble = ensemble
        self.half = half
        self.ncnn_gpu = ncnn_gpu
        self.threads = threads
        self.handleMethod()

    def handleMethod(self):
        if "rife-ncnn-python" == self.method:
            self.interpolate_process = RifeNCNN(
                interpolation_factor=self.interpolation_factor,
                interpolate_method=self.model,
                width=self.originalWidth,
                height=self.originalHeight,
                ensemble=self.ensemble,
                half=self.half,
                ncnn_gpu=self.ncnn_gpu,
            )
        if "rife-cuda" == self.method:
            self.interpolate_process = Rife(
                interpolation_factor=self.interpolation_factor,
                interpolate_method=self.model,
                width=self.originalWidth,
                height=self.originalHeight,
                ensemble=self.ensemble,
                half=self.half,
            )
        if "rife-cuda-trt" == self.method:
            self.interpolate_process = RifeTensorRT(
                model=self.model,
                width=self.originalWidth,
                height=self.originalHeight,
                ensemble=self.ensemble,
                half=self.half,
                guiLog=self.guiLog,
            )
        if "gmfss" in self.model:
            self.interpolate_process = GMFSS(
                interpolation_factor=self.interpolation_factor,
                width=self.originalWidth,
                height=self.originalHeight,
                ensemble=self.ensemble,
                half=self.half,
            )
        self.main.start_time = time.time()
        print("starting render")

    def proc_image(self, frame0, frame1):
        self.interpolate_process.run1(frame0, frame1)

        self.frame += 1
        self.writeBuffer.put(frame0)

        for i in range(self.interpolation_factor - 1):
            result = self.interpolate_process.make_inference(
                (i + 1) * 1.0 / (self.interpolation_factor)
            )
            self.frame += 1
            self.writeBuffer.put(result)

    def procInterpThread(self):
        self.frame = 0

        while True:
            if self.main.settings.SceneChangeDetectionMode == "Enabled":
                if len(self.main.transitionFrames) > 0:
                    self.transition_frame = self.main.transitionFrames[0]
                else:
                    self.transition_frame = -1
            else:
                self.transition_frame = -1
            frame = self.readBuffer.get()

            if frame is None:
                log("done with proc")
                self.writeBuffer.put(self.prevFrame)
                self.writeBuffer.put(None)
                break  # done with proc

            if self.prevFrame is None:
                self.prevFrame = frame
                continue

            if self.frame == self.transition_frame - self.interpolation_factor:
                for i in range(self.interpolation_factor):
                    self.writeBuffer.put(frame)
                self.frame += self.interpolation_factor
                self.transition_frame = self.main.transitionFrames.pop(0)
            else:
                self.proc_image(self.prevFrame, frame)

            self.prevFrame = frame


class Upscaling(Render):
    def __init__(
        self,
        main,
        input_file,
        output_file,
        resIncrease,
        model_path,
        half,
        method="cuda",
        threads=2,
        cugan_noise=0,
        ncnn_gpu=0,
        benchmark=False,
        modelName="",
        guiLog=None,
    ):
        super(Upscaling, self).__init__(
            main,
            input_file,
            output_file,
            interpolationIncrease=1,
            resIncrease=resIncrease,
            benchmark=benchmark,
        )
        self.model_path = model_path
        self.half = half
        self.method = method
        self.resIncrease = resIncrease
        self.threads = threads
        self.frame = 0
        self.cugan_noise = cugan_noise
        self.ncnn_gpu = ncnn_gpu
        self.modelName = modelName
        self.guiLog = guiLog
        self.handleModel()

    def handleModel(self):
        if "cuda" in self.method and "ncnn" not in self.method:
            self.upscaleMethod = UpscaleCUDA(
                self.originalWidth, self.originalHeight, self.model_path, self.half
            )
        if "tensorrt" in self.method and "ncnn" not in self.method:
            self.upscaleMethod = UpscaleTensorRT(
                width=self.originalWidth,
                height=self.originalHeight,
                modelPath=self.model_path,
                half=self.half,
                modelName=self.modelName,
                guiLog=self.guiLog,
                upscaleFactor=self.resIncrease,
            )
        if "ncnn" in self.method and not "cugan" in self.method:
            self.upscaleMethod = UpscaleNCNN(
                gpuid=self.ncnn_gpu,
                model=self.model_path,
                num_threads=self.threads,
                scale=self.resIncrease,
                width=self.originalWidth,
                height=self.originalHeight,
            )
        if "cugan" in self.method and "ncnn" in self.method:
            self.upscaleMethod = UpscaleCuganNCNN(
                gpuid=self.ncnn_gpu,
                models_path=os.path.join(
                    f"{thisdir}", "models", "realcugan", "models-se"
                ),
                model="models-se",
                num_threads=self.threads,
                scale=self.resIncrease,
                width=self.originalWidth,
                height=self.originalHeight,
            )
        self.main.start_time = time.time()

    def procUpscaleThread(self):
        while True:
            frame = self.readBuffer.get()

            if frame is None:
                log("done with proc")
                self.writeBuffer.put(None)
                break  # done with proc

            result = self.upscaleMethod.UpscaleImage(frame)

            self.writeBuffer.put(result)
            self.prevFrame = result

            self.frame += 1
