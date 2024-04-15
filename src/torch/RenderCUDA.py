import subprocess
import numpy as np
from queue import Queue

from .rife.rife import *
from src.programData.thisdir import thisdir

thisdir = thisdir()
import sys
from threading import Thread
import cv2
import re
from src.programData.settings import *
import src.programData.return_data as return_data
from time import sleep
from .UpscaleImage import UpscaleCUDA
try:
    from src.torch.gmfss.gmfss_fortuna_union import GMFSS
except:
    pass
# read
# Calculate eta by time remaining divided by speed
# add scenedetect by if frame_num in transitions in proc_frames
# def
class Render:
    def __init__(
        self, main, input_file, output_file, interpolationIncrease=1, resIncrease=1
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

        self.settings = Settings()

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

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.frame_size = self.originalWidth * self.originalHeight * 3

    def readThread(self):
        while True:
            chunk = self.process.stdout.read(self.frame_size)
            if len(chunk) < self.frame_size:
                self.process.stdout.close()
                self.process.terminate()
                self.readingDone = True
                self.readBuffer.put(None)
                log("done with read")
                break
            frame = np.frombuffer(chunk, dtype=np.uint8).reshape(
                (self.originalHeight, self.originalWidth, 3)
            )

            self.readBuffer.put(frame)

    def finish_render(self):
        self.writeBuffer.put(None)

    def returnLatestFrame(self):
        if self.prevFrame:
            return self.prevFrame

    def returnFrameCount(self):
        try:
            return self.frame
        except:
            log("No frame to return!")

    def returnFrameRate(self):
        try:
            return self.frameRate
        except:
            log("No framerate to return!")

    def returnPercentageDone(self):
        try:
            return self.frame / self.frame_count
        except:  # noqa: E722
            log("No frame to return!")

    def log(self):
        while not self.main.CudaRenderFinished:
            try:
                for line in iter(self.writeProcess.stderr.readline, b""):
                    if not self.main.CudaRenderFinished:
                        log(line)
                        # print(line)
                    else:
                        break
                    # self.frame = re.findall(r'frame=\d+',line.replace(' ',''))[0].replace('frame=','')
                    # self.frame = int(self.frame.replace('frame=',''))

                    # self.frameRate = int(re.findall(r'frame=\d+',line.replace(' ','')))[0].replace('fps=','')

            except Exception as e:
                pass
                # tb = traceback.format_exc()
                # log(tb,e)
                # log(f'{tb},{e}')
        log("Done with logging")
        log("done with logging thread for cuda")

    # save

    def FFmpegOut(self):
        log("saving")
        try:
            crf = return_data.returnCRFFactor(
                self.settings.videoQuality, self.settings.Encoder
            )
        except Exception as e:
            log(f"unable to set crf {e}")

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
            try:
                frame = self.writeBuffer.get()
                if frame is None:
                    self.writeProcess.stdin.close()
                    self.writeProcess.wait()
                    log("done with save")
                    self.main.output_file = self.output_file
                    self.main.CudaRenderFinished = True
                    torch.cuda.empty_cache()
                    break

                frame = np.ascontiguousarray(frame)
                self.main.imageDisplay = frame
                self.writeProcess.stdin.buffer.write(frame.tobytes())
            except Exception as e:
                tb = traceback.format_exc()
                log(f"Something went wrong with the writebuffer: {e},{tb}")


class Interpolation(Render):
    def __init__(self, main, method, input_file, output_file, model, times, ensemble, half):
        super(Interpolation, self).__init__(
            main, input_file, output_file, interpolationIncrease=times, resIncrease=1
        )
        self.method = method
        self.model = model
        self.ensemble = ensemble
        self.half = half
        self.handleMethod()
    def handleMethod(self):
        if 'rife' in self.model:
            self.interpolate_process = Rife(
                interpolation_factor=self.interpolation_factor,
                interpolate_method=self.model,
                width=self.originalWidth,
                height=self.originalHeight,
                ensemble=self.ensemble,
                half=self.half,
            )
        if 'gmfss' in self.model:
            self.interpolate_process = GMFSS(
                interpolation_factor=self.interpolation_factor,
                width=self.originalWidth,
                height=self.originalHeight,
                ensemble=self.ensemble,
                half=self.half,
            )
    
    def proc_image(self, frame0,frame1):
        self.interpolate_process.run1(frame0,frame1)
        
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
                self.proc_image(self.prevFrame,frame)

            self.prevFrame = frame


class Upscaling(Render):
    def __init__(self, main, input_file, output_file, resIncrease, model_path, half):
        super(Upscaling, self).__init__(
            main,
            input_file,
            output_file,
            interpolationIncrease=1,
            resIncrease=resIncrease,
        )
        self.model_path = model_path
        self.half = half

    def procUpscaleThread(self):
        self.frame = 0
        self.upscaleMethod = UpscaleCUDA(
            self.originalWidth, self.originalHeight, self.model_path, self.half
        )

        while True:
            frame = self.readBuffer.get()

            if frame is None:
                log("done with proc")
                self.writeBuffer.put(frame)
                self.writeBuffer.put(None)
                break  # done with proc

            result = self.upscaleMethod.UpscaleImage(frame)

            self.writeBuffer.put(result)
            self.prevFrame = result

            self.frame += 1