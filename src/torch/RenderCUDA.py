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
from spandrel import ImageModelDescriptor, ModelLoader

# read
# Calculate eta by time remaining divided by speed
# add scenedetect by if frame_num in transitions in proc_frames
# def
class Render:
    def __init__(self, main, input_file, output_file, interpolationIncrease=1, resIncrease=1):
        self.main = main

        self.readBuffer = Queue(maxsize=50)
        self.writeBuffer = Queue(maxsize=50)
        self.interpolation_factor = round(interpolationIncrease)
        self.prevFrame = None
        
        
        
        self.input_file = input_file
        self.output_file = output_file
        
        cap = cv2.VideoCapture(input_file)
        self.initialFPS = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * resIncrease
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * resIncrease
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
            f"{self.width}x{self.height}",
            "-",
        ]
        

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.frame_size = self.width * self.height * 3

    def readThread(self):
        while True:
            chunk = self.process.stdout.read(self.frame_size)
            if len(chunk) < self.frame_size:
                self.process.stdout.close()
                self.process.terminate()
                self.readingDone = True
                self.readBuffer.put(None)
                print("done with read")
                break
            frame = np.frombuffer(chunk, dtype=np.uint8).reshape(
                (self.height, self.width, 3)
            )

            self.readBuffer.put(frame)

        
    def finish_render(self):
        self.writeBuffer.put(None)

    def returnLatestFrame(self):
        try:
            return self.prevFrame
        except:
            print("No frame to return!")

    def returnFrameCount(self):
        try:
            return self.frame
        except:
            print("No frame to return!")

    def returnFrameRate(self):
        try:
            return self.frameRate
        except:
            print("No framerate to return!")

    def returnPercentageDone(self):
        try:
            return self.frame / self.frame_count
        except:  # noqa: E722
            print("No frame to return!")

    def log(self):
        sleep(1)
        while not self.main.CudaRenderFinished:
            try:
                for line in iter(self.writeProcess.stderr.readline, b""):
                    log(line)
                    # self.frame = re.findall(r'frame=\d+',line.replace(' ',''))[0].replace('frame=','')
                    # self.frame = int(self.frame.replace('frame=',''))

                    # self.frameRate = int(re.findall(r'frame=\d+',line.replace(' ','')))[0].replace('fps=','')

            except Exception as e:
                pass
                # tb = traceback.format_exc()
                # print(tb,e)
                # log(f'{tb},{e}')
        print("Done with logging")
        log("done with logging thread for cuda")

    # save

    def FFmpegOut(self):
        print("saving")
        try:
            crf = return_data.returnCRFFactor(
                self.settings.videoQuality, self.settings.Encoder
            )
        except Exception as e:
            print(f"unable to set crf {e}")

        command = [
            f"{thisdir}/bin/ffmpeg",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{int(self.finalFPS)}",
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
        print(command)
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
                    print("done with save")
                    self.main.output_file = self.output_file
                    self.main.CudaRenderFinished = True
                    break
                self.main.imageDisplay = frame
                frame = np.ascontiguousarray(frame)

                self.writeProcess.stdin.buffer.write(frame.tobytes())
            except Exception as e:
                tb = traceback.format_exc()
                print(f"Something went wrong with the writebuffer: {e},{tb}")

class Interpolation(Render):
    
    def __init__(self, main, input_file, output_file, times):
        super(Interpolation, self).__init__(main, input_file, output_file, interpolationIncrease=times,resIncrease=1)
        self.interpolate_process = Rife(
            interpolation_factor=self.interpolation_factor,
            interpolate_method="rife4.14",
            width=self.width,
            height=self.height,
            half=True,
        )
    def proc_image(self, frame1, frame2):
        self.interpolate_process.run(frame1, frame2)

        self.writeBuffer.put(frame1)
        self.frame += 1
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
                print("done with proc")
                self.writeBuffer.put(self.prevFrame)
                self.writeBuffer.put(None)
                break  # done with proc

            if self.prevFrame is None:
                self.prevFrame = frame
                continue

            if self.frame != self.transition_frame - self.interpolation_factor:
                self.proc_image(self.prevFrame, frame)

            else:
                for i in range(self.interpolation_factor):
                    self.writeBuffer.put(frame)

                self.frame += self.interpolation_factor
                self.transition_frame = self.main.transitionFrames.pop(0)

            self.prevFrame = frame

class Upscaling(Render):
    
    def __init__(self, main, input_file, output_file, resIncrease):
        super(Upscaling, self).__init__(main, input_file, output_file, interpolationIncrease=1,resIncrease=resIncrease)
        self.model = ModelLoader().load_from_file(f"{thisdir}/models/realesrgan-cuda/realesr-animevideov3.pth")

        assert isinstance(self.model, ImageModelDescriptor)

        self.model.eval() # gonna have to put cuda back in here lmfaooooooo
        
    def proc_image(self, image):
        with torch.no_grad():
            self.writeBuffer.put(self.model(image))

    def procUpscaleThread(self):
        self.frame = 0

        while True:
            
            frame = self.readBuffer.get()

            if frame is None:
                print("done with proc")
                self.writeBuffer.put(frame)
                self.writeBuffer.put(None)
                break  # done with proc

            
            self.proc_image(frame)
            

            

            self.frame += 1

            
