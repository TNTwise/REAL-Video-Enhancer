import cv2
import os
import subprocess
import queue
import time

from .Util import currentDirectory


class FFMpegRender:
    def __init__(
        self,
        inputFile: str,
        outputFile: str,
        interpolateFactor: int = 1,
        upscaleTimes: int = 1,
        encoder: str = "libx264",
        pixelFormat: str = "yuv420p",
        benchmark: bool = False,
        overwrite: bool = False,
        frameSetupFunction=None,
        crf: str = "18",
    ):
        """
        Generates FFmpeg I/O commands to be used with VideoIO
        Options:
        inputFile: str, The path to the input file.
        outputFile: str, The path to the output file.
        interpolateTimes: int, this sets the multiplier for the framerate when interpolating, when only upscaling this will be set to 1.
        upscaleTimes: int,
        encoder: str, The exact name of the encoder ffmpeg will use (default=libx264)
        pixelFormat: str, The pixel format ffmpeg will use, (default=yuv420p)
        overwrite: bool, overwrite existing output file if it exists
        """
        self.inputFile = inputFile
        self.outputFile = outputFile

        # upsacletimes will be set to the scale of the loaded model with spandrel
        self.upscaleTimes = upscaleTimes
        self.interpolateFactor = interpolateFactor
        self.encoder = encoder
        self.pixelFormat = pixelFormat
        self.benchmark = benchmark
        self.overwrite = overwrite
        self.readingDone = False
        self.writeOutPipe = False
        self.crf = crf
        self.frameSetupFunction = frameSetupFunction

        self.writeOutPipe = self.outputFile == "PIPE"
        self.totalFramesToRender = (
            self.totalFrames - self.interpolateFactor
        ) * self.interpolateFactor

        self.readQueue = queue.Queue(maxsize=50)
        self.writeQueue = queue.Queue(maxsize=50)

    def getVideoProperties(self, inputFile: str = None):
        if inputFile is None:
            cap = cv2.VideoCapture(self.inputFile)
        else:
            cap = cv2.VideoCapture(inputFile)
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        self.frameChunkSize = self.width * self.height * 3

    def getFFmpegReadCommand(self):
        command = [
            f"{os.path.join(currentDirectory(),'bin','ffmpeg')}",
            "-i",
            f"{self.inputFile}",
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
        return command

    def getFFmpegWriteCommand(self):
        if not self.outputFile == "PIPE":
            if not self.benchmark:
                # maybe i can split this so i can just use ffmpeg normally like with vspipe
                command = [
                    f"{os.path.join(currentDirectory(),'bin','ffmpeg')}",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-vcodec",
                    "rawvideo",
                    "-s",
                    f"{self.width * self.upscaleTimes}x{self.height * self.upscaleTimes}",
                    "-r",
                    f"{self.fps * self.interpolateFactor}",
                    "-i",
                    "-",
                    "-i",
                    f"{self.inputFile}",
                    f"-crf",
                    f"{self.crf}",
                    "-pix_fmt",
                    self.pixelFormat,
                    "-c:a",
                    "copy",
                    f"{self.outputFile}",
                ]
                for i in self.encoder.split():
                    command.append(i)
            else:
                print("Using benchmark mode")
                command = [
                    f"{os.path.join(currentDirectory(),'bin','ffmpeg')}",
                    "-y",
                    "-v",
                    "warning",
                    "-stats",
                    "-f",
                    "rawvideo",
                    "-vcodec",
                    "rawvideo",
                    "-s",
                    f"{self.width * self.upscaleTimes}x{self.height * self.upscaleTimes}",
                    "-pix_fmt",
                    f"yuv420p",
                    "-r",
                    f"{self.fps * self.interpolateFactor}",
                    "-i",
                    "-",
                    "-benchmark",
                    "-f",
                    "null",
                    "-",
                ]
            if self.overwrite:
                command.append("-y")
            return command

    def readinVideoFrames(self):
        self.readProcess = subprocess.Popen(
            self.getFFmpegReadCommand(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        for i in range(self.totalFrames - 1):
            chunk = self.readProcess.stdout.read(self.frameChunkSize)
            self.readQueue.put(chunk)
        self.readQueue.put(None)
        self.readingDone = True
        self.readProcess.stdout.close()
        self.readProcess.terminate()

    def returnFrame(self, frame):
        return frame

    def writeOutVideoFrames(self):
        """
        Writes out frames either to ffmpeg or to pipe
        This is determined by the --output command, which if the PIPE parameter is set, it outputs the chunk to pipe.
        A command like this is required,
        ffmpeg -f rawvideo -pix_fmt rgb24 -s 1920x1080 -framerate 24 -i - -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a copy out.mp4
        """
        startTime = time.time()
        if self.writeOutPipe == False:
            self.writeProcess = subprocess.Popen(
                self.getFFmpegWriteCommand(),
                stdin=subprocess.PIPE,
                text=True,
                universal_newlines=True,
            )

            while True:
                frame = self.writeQueue.get()
                if frame is None:
                    break
                self.writeProcess.stdin.buffer.write(frame)
            self.writeProcess.stdin.close()
            self.writeProcess.wait()

        else:
            process = subprocess.Popen(["cat"], stdin=subprocess.PIPE)
            while True:
                frame = self.writeQueue.get()
                if frame is None:
                    break
                process.stdin.write(frame)
            process.stdin.close()
            process.wait()
        
        renderTime = time.time() - startTime
        print(f"Time to complete render: {round(renderTime, 2)}")
