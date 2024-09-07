import cv2
import re
import os
import subprocess
import queue
import sys
import time
import math
from tqdm import tqdm
from multiprocessing import shared_memory
from .Util import currentDirectory, log, printAndLog, ffmpegPath
import time
from time import sleep
from threading import Thread

def convertTime(remaining_time):
    """
    Converts seconds to hours, minutes and seconds
    """
    hours = remaining_time // 3600
    remaining_time -= 3600 * hours
    minutes = remaining_time // 60
    remaining_time -= minutes * 60
    seconds = remaining_time
    if minutes < 10:
        minutes = str(f"0{minutes}")
    if seconds < 10:
        seconds = str(f"0{seconds}")
    return hours, minutes, seconds


class FFMpegRender:
    """Args:
        inputFile (str): The path to the input file.
        outputFile (str): The path to the output file.
        interpolateFactor (int, optional): Sets the multiplier for the framerate when interpolating. Defaults to 1.
        upscaleTimes (int, optional): Upscaling factor. Defaults to 1.
        encoder (str, optional): The exact name of the encoder ffmpeg will use. Defaults to "libx264".
        pixelFormat (str, optional): The pixel format ffmpeg will use. Defaults to "yuv420p".
        benchmark (bool, optional): Enable benchmark mode. Defaults to False.
        overwrite (bool, optional): Overwrite existing output file if it exists. Defaults to False.
        frameSetupFunction (function, optional): Function to setup frames. Defaults to None.
        crf (str, optional): Constant Rate Factor for video quality. Defaults to "18".
        sharedMemoryID (str, optional): ID for shared memory. Defaults to None.
        shm (shared_memory.SharedMemory, optional): Shared memory object. Defaults to None.
        inputFrameChunkSize (int, optional): Size of input frame chunks. Defaults to None.
        outputFrameChunkSize (int, optional): Size of output frame chunks. Defaults to None.
    pass
    Gets the properties of the video file.
    Args:
        inputFile (str, optional): The path to the input file. If None, uses the inputFile specified in the constructor. Defaults to None.
    pass
    Generates the FFmpeg command for reading video frames.
    Returns:
        list: The FFmpeg command for reading video frames.
    pass
    Generates the FFmpeg command for writing video frames.
    Returns:
        list: The FFmpeg command for writing video frames.
    pass
    Starts reading video frames using FFmpeg.
    pass
    Returns a frame.
    Args:
        frame: The frame to be returned.
    Returns:
        The returned frame.
    pass
    Prints data in real-time.
    Args:
        data: The data to be printed.
    pass
    Writes frames to shared memory.
    Args:
        fcs: The frame chunk size.
    pass
    Writes out video frames using FFmpeg.
    pass"""

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
        sharedMemoryID: str = None,
        channels=3,

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
        self.ceilInterpolateFactor = math.ceil(self.interpolateFactor)
        self.encoder = encoder
        self.pixelFormat = pixelFormat
        self.benchmark = benchmark
        self.overwrite = overwrite
        self.readingDone = False
        self.writingDone = False
        self.writeOutPipe = False
        self.previewFrame = None
        self.crf = crf
        self.frameSetupFunction = frameSetupFunction
        self.sharedMemoryID = sharedMemoryID
        self.videoPropertiesLocation = os.path.join(currentDirectory(), inputFile + "_VIDEODATA")
        if not os.path.exists(self.videoPropertiesLocation):
            os.makedirs(self.videoPropertiesLocation)
        self.subtitleFiles = []
        self.sharedMemoryThread = Thread(
            target=lambda: self.writeOutInformation(self.outputFrameChunkSize)
        )
        self.inputFrameChunkSize = self.width * self.height * channels
        self.outputFrameChunkSize = (
            self.width * self.upscaleTimes * self.height * self.upscaleTimes * channels
        )
        self.shm = shared_memory.SharedMemory(
            name=self.sharedMemoryID, create=True, size=self.outputFrameChunkSize
        )
        self.totalOutputFrames = self.totalInputFrames * self.ceilInterpolateFactor

        self.writeOutPipe = self.outputFile == "PIPE"

        self.readQueue = queue.Queue(maxsize=50)
        self.writeQueue = queue.Queue(maxsize=50)
    

    def get_ffmpeg_streams(self,video_file):
        """Get a list of streams from the video file using FFmpeg."""
        try:
            result = subprocess.run(
                [ffmpegPath(), '-i', video_file],
                stderr=subprocess.PIPE,
                text=True
            )
            return result.stderr
        except Exception as e:
            print(f"An error occurred while running FFmpeg: {e}")
            return None

    def extract_subtitles(self,video_file, stream_index, subtitle_file):
        """Extract a specific subtitle stream from the video file."""
        try:
            subprocess.run(
                [ffmpegPath(), '-i', video_file, '-map', f'0:{stream_index}', subtitle_file],
                check=True
            )
            print(f"Extracted subtitle stream {stream_index} to {subtitle_file}")
            self.subtitleFiles.append(subtitle_file)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while extracting subtitles: {e}")

    def getVideoSubs(self, video_file):


        ffmpeg_output = self.get_ffmpeg_streams(video_file)
        if not ffmpeg_output:
            return

        subtitle_stream_pattern = re.compile(r'Stream #0:(\d+).*?Subtitle', re.MULTILINE | re.DOTALL)
        subtitle_streams = subtitle_stream_pattern.findall(ffmpeg_output)

        if not subtitle_streams:
            print("No subtitle streams found in the video.")
            return

        for stream_index in subtitle_streams:
            subtitle_file = os.path.join(self.videoPropertiesLocation, f'subtitle_{stream_index}.srt')
            self.extract_subtitles(video_file, stream_index, subtitle_file)


    def getVideoProperties(self, inputFile: str = None):
        log("Getting Video Properties...")
        if inputFile is None:
            cap = cv2.VideoCapture(self.inputFile)
        else:
            cap = cv2.VideoCapture(inputFile)
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.totalInputFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        self.outputFrameChunkSize = None

    def getFFmpegReadCommand(self):
        log("Generating FFmpeg READ command...")
        command = [
            f"{ffmpegPath()}",
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
        log("Generating FFmpeg WRITE command...")
        if not self.benchmark:
            # maybe i can split this so i can just use ffmpeg normally like with vspipe
            command = [
                f"{ffmpegPath()}",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{self.width * self.upscaleTimes}x{self.height * self.upscaleTimes}",
                "-r",
                f"{self.fps * self.ceilInterpolateFactor}",
                "-i",
                "-",
                "-i",
                f"{self.inputFile}",
                #"""["-i" + subtitle for subtitle in self.subtitleFiles],"""
                "-r",
                f"{self.fps * self.interpolateFactor}",
                f"-crf",
                f"{self.crf}",
                "-pix_fmt",
                self.pixelFormat,
                "-c:a",
                "copy",
                "-loglevel",
                "error",
            ]
            for i in self.encoder.split():
                command.append(i)
            
            command.append(
                f"{self.outputFile}",
            )

            if self.overwrite:
                command.append("-y")
            return command

    def readinVideoFrames(self):
        log("Starting Video Read")
        self.readProcess = subprocess.Popen(
            self.getFFmpegReadCommand(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        for i in range(self.totalInputFrames - 1):
            chunk = self.readProcess.stdout.read(self.inputFrameChunkSize)
            self.readQueue.put(chunk)
        log("Ending Video Read")
        self.readQueue.put(None)
        self.readingDone = True
        self.readProcess.stdout.close()
        self.readProcess.terminate()

    def returnFrame(self, frame):
        return frame

    def realTimePrint(self, data):
        data = str(data)
        # Clear the last line
        sys.stdout.write("\r" + " " * self.last_length)
        sys.stdout.flush()

        # Write the new line
        sys.stdout.write("\r" + data)
        sys.stdout.flush()

        # Update the length of the last printed line
        self.last_length = len(data)

    def calculateETA(self):
        """
        Calculates ETA

        Gets the time for every frame rendered by taking the
        elapsed time / completed iterations (files)
        remaining time = remaining iterations (files) * time per iteration

        """

        # Estimate the remaining time
        elapsed_time = time.time() - self.startTime
        time_per_iteration = elapsed_time / self.framesRendered
        remaining_iterations = self.totalOutputFrames - self.framesRendered
        remaining_time = remaining_iterations * time_per_iteration
        remaining_time = int(remaining_time)
        # convert to hours, minutes, and seconds
        hours, minutes, seconds = convertTime(remaining_time)
        return f"{hours}:{minutes}:{seconds}"

    def writeOutInformation(self, fcs):
        """
        fcs = framechunksize
        """
        # Create a shared memory block

        buffer = self.shm.buf

        log(f"Shared memory name: {self.shm.name}")
        while True:
            if self.writingDone:
                self.shm.close()
                self.shm.unlink()
                break
            if self.previewFrame is not None:
                # print out data to stdout
                fps = round(self.framesRendered / (time.time() - self.startTime))
                eta = self.calculateETA()
                message = f"FPS: {fps} Current Frame: {self.framesRendered} ETA: {eta}"
                self.realTimePrint(message)
                if self.sharedMemoryID is not None:
                    # Update the shared array
                    buffer[:fcs] = bytes(self.previewFrame)

            time.sleep(0.1)

    def writeOutVideoFrames(self):
        """
        Writes out frames either to ffmpeg or to pipe
        This is determined by the --output command, which if the PIPE parameter is set, it outputs the chunk to pipe.
        A command like this is required,
        ffmpeg -f rawvideo -pix_fmt rgb24 -s 1920x1080 -framerate 24 -i - -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a copy out.mp4
        """
        log("Rendering")
        #
        self.startTime = time.time()
        self.framesRendered: int = 0
        self.last_length: int = 0

        if self.benchmark:
            while True:
                frame = self.writeQueue.get()
                self.previewFrame = frame
                if frame is None:
                    break
                self.framesRendered += 1
        else:
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
                # Update other variables
                self.previewFrame = frame
                # Update progress bar
                # pbar.update(1)
                self.framesRendered += 1
            self.writeProcess.stdin.close()
            self.writeProcess.wait()

        renderTime = time.time() - self.startTime
        self.writingDone = True
        printAndLog(f"\nTime to complete render: {round(renderTime, 2)}")
