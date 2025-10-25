import queue
import sys
from abc import ABC, abstractmethod
import os
import subprocess
import queue
import time
import cv2
import numpy as np

from .constants import FFMPEG_PATH, FFMPEG_LOG_FILE
from .utils.Util import (
    log,
    subprocess_popen_without_terminal,
)
from .utils.Encoders import  EncoderSettings

class Buffer(ABC):
    @abstractmethod
    def command(self) -> list[str]:
        pass


class FFmpegRead(Buffer):
    def __init__(self, inputFile, width, height, start_time, end_time, borderX, borderY, hdr_mode, color_space=None, color_primaries=None, color_transfer=None, input_pixel_format: str | None = None):
        self.inputFile = inputFile
        self.width = width
        self.height = height
        self.start_time = start_time
        self.end_time = end_time
        self.borderX = borderX
        self.borderY = borderY
        self.hdr_mode = hdr_mode
        self.color_space = color_space
        self.color_primaries = color_primaries
        self.color_transfer = color_transfer
        self.input_pixel_format = input_pixel_format
        self.yuv420pMOD = self.input_pixel_format == "yuv420p" and not self.hdr_mode
        #self.yuv420pMOD = False
        if self.hdr_mode:
            self.inputFrameChunkSize = width * height * 6
        else:
            if self.yuv420pMOD:
                self.inputFrameChunkSize = width * height * 3 // 2
            else:
                self.inputFrameChunkSize = width * height * 3
        command = self.command()
        log("FFMPEG READ COMMAND: " + str(command))
        self.readProcess = subprocess_popen_without_terminal(
            self.command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.readQueue = queue.Queue(maxsize=25)

    def command(self):
        
        command = [
            f"{FFMPEG_PATH}",
            "-i",
            f"{self.inputFile}",
        ]

        filter_string = f"crop={self.width}:{self.height}:{self.borderX}:{self.borderY},scale=w=iw*sar:h=ih" #+ ":in_range=limited:out_range=full,format=yuv420p" if self.yuv420pMOD == "yuv420p" else "" # fix dar != sar
        #if not self.hdr_mode:
        #    if self.input_pixel_format == "yuv420p":
        #        filter_string += ":in_range=tv:out_range=pc" # color shifts a smidgen but helps with artifacts when converting yuv to raw
        command += [
            "-vf",
            filter_string,
            "-f",
            "image2pipe",
            "-pix_fmt",
            "rgb48le" if self.hdr_mode else (self.input_pixel_format if self.yuv420pMOD else "rgb24"),
            #"rgb48le" if self.hdr_mode else "rgb24",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{self.width}x{self.height}",
            "-"
            
        ]

        log("FFMPEG READ COMMAND: " + str(command))
        return command

    def read_frame(self):
        chunk = self.readProcess.stdout.read(self.inputFrameChunkSize)
        if len(chunk) < self.inputFrameChunkSize:
            return None

        if self.yuv420pMOD:
            # Convert raw YUV420p data to RGB
            # The data is Y plane, then U plane, then V plane, concatenated.
            # cv2.COLOR_YUV420P2RGB expects a single channel image of shape (height * 3 // 2, width)
            np_frame = np.frombuffer(chunk, dtype=np.uint8)
            # Ensure height is an integer for reshape, Python 3 // operator already does this.
            yuv_image_height = self.height * 3 // 2
            yuv_image = np_frame.reshape((yuv_image_height, self.width))
            rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB_I420)
            # cv2.imwrite("temp_rgb_image.png", rgb_image)  # Debugging line, can be removed
            chunk = rgb_image.tobytes()
        
        return chunk

    def read_frames_into_queue(self):
        while True:
            chunk = self.read_frame()
            if chunk is None:
                break
            self.readQueue.put(chunk)
        self.readQueue.put(None)

    def get(self):
        return self.readQueue.get()

    def close(self):
        self.readProcess.stdout.close()
        self.readProcess.terminate()


class FFmpegWrite(Buffer):
    def __init__(
        self,
        inputFile: str,
        outputFile: str,
        width: int,
        height: int,
        start_time: float,
        end_time: float,
        fps: float,
        crf: str,
        audio_bitrate: str,
        pixelFormat: str,
        overwrite: bool,
        custom_encoder: str,
        benchmark: bool,
        slowmo_mode: bool,
        upscaleTimes: int,
        interpolateFactor:int,
        ceilInterpolateFactor: int,
        video_encoder: EncoderSettings,
        audio_encoder: EncoderSettings,
        subtitle_encoder: EncoderSettings,
        hdr_mode: bool,
        mpv_output: bool,
        merge_subtitles: bool,
        color_space: str = None,
        color_primaries: str = None,
        color_transfer: str = None,
    ):
        self.inputFile = inputFile
        self.outputFile = outputFile
        if self.outputFile:
            self.outputFileExtension = os.path.split(self.outputFile)[-1].split(".")[-1]
        self.width = width
        self.height = height
        self.start_time = start_time
        self.end_time = end_time
        self.outputWidth = width * upscaleTimes
        self.outputHeight = height * upscaleTimes
        self.fps = fps
        self.crf = crf
        self.audio_bitrate = audio_bitrate
        self.pixelFormat = pixelFormat
        self.overwrite = overwrite
        self.custom_encoder = custom_encoder
        self.benchmark = benchmark
        self.slowmo_mode = slowmo_mode
        self.upscaleTimes = upscaleTimes
        self.interpolateFactor = interpolateFactor
        self.ceilInterpolateFactor = ceilInterpolateFactor
        self.video_encoder = video_encoder
        self.audio_encoder = audio_encoder
        self.subtitle_encoder = subtitle_encoder
        self.mpv_output = mpv_output
        self.hdr_mode = hdr_mode
        self.writeQueue = queue.Queue(maxsize=25)
        self.previewFrame = None
        self.framesRendered: int = 1
        self.writeProcess = None
        self.color_space = color_space
        self.color_primaries = color_primaries
        self.color_transfer = color_transfer
        self.outputFPS = (
            (self.fps * self.interpolateFactor)
            if not self.slowmo_mode
            else self.fps
        )
        self.ffmpeg_log = open(FFMPEG_LOG_FILE, "w", encoding='utf-8')
        try:
            command = self.command()
            log("\nFFMPEG WRITE COMMAND: " + str(command) + "\n")
            self.writeProcess = subprocess_popen_without_terminal(
                command,
                stdin=subprocess.PIPE,
                stderr=self.ffmpeg_log,
                stdout=subprocess.PIPE if self.mpv_output else self.ffmpeg_log,
                text=True,
                universal_newlines=True,
            )
        except Exception as e:
            self.onErroredExit()


    def command(self):
        if self.mpv_output:
            command = [
                f"{FFMPEG_PATH}",
                "-loglevel",
                "error",
                "-framerate",
                f"{self.outputFPS}",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb48le" if self.hdr_mode else "rgb24",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{self.outputWidth}x{self.outputHeight}",
                "-i",
                "-",
                "-r",
                f"{self.outputFPS}",
                "-f",
                "matroska",
                "-b:v",
                "15000k",
                "-crf",
                "0",
                "-af",
                f"atrim=start={self.start_time},asetpts=PTS-STARTPTS",

            ]
            
            
            if self.hdr_mode:

                # override pixel format
                pxfmtdict = {
                    "yuv420p": "yuv420p10le",
                    "yuv422": "yuv422p10le",
                    "yuv444": "yuv444p10le",
                }

                if self.pixelFormat in pxfmtdict:
                    self.pixelFormat = pxfmtdict[self.pixelFormat]
                
                command += [
                    "-pix_fmt",
                    self.pixelFormat,
                ]
                
            command += [
                "-",
            ]
            return command


        if not self.benchmark:
            # maybe i can split this so i can just use ffmpeg normally like with vspipe
            command = [
                f"{FFMPEG_PATH}",
                "-loglevel",
                "error",
            ]

            if self.custom_encoder is None:
                pre_in_set = self.video_encoder.getPreInputSettings()
                if pre_in_set is not None:
                    command += pre_in_set.split()

            command += [
                "-framerate",
                f"{self.outputFPS}",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb48le" if self.hdr_mode else "rgb24",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{self.outputWidth}x{self.outputHeight}",
                "-i",
                "-",
                "-r",
                f"{self.outputFPS}",
            ]

            if not self.slowmo_mode:
                command += [
                    "-i",
                    f"{self.inputFile}",
                    "-map",
                    "0:v",  # Map video stream from input 0
                    "-map",
                    "1:a?",

                ]

                

            if self.custom_encoder is not None:

                for i in self.custom_encoder.split():
                    command.append(i)
            else:
                if not self.audio_encoder.getPresetTag() == "copy_audio":
                    command += [
                        "-b:a",
                        self.audio_bitrate,
                    ]
                command += self.video_encoder.getPostInputSettings().split()
                command += [self.video_encoder.getQualityControlMode(), str(self.crf)]
                command += self.audio_encoder.getPostInputSettings().split()
                command += self.subtitle_encoder.getPostInputSettings().split()

                if self.hdr_mode:
                    
                    

                    # override pixel format
                    pxfmtdict = {
                        "yuv420p": "yuv420p10le",
                        "yuv422": "yuv422p10le",
                        "yuv444": "yuv444p10le",
                    }

                    if self.pixelFormat in pxfmtdict:
                        self.pixelFormat = pxfmtdict[self.pixelFormat]

                    if self.video_encoder.getPresetTag() == "libx265" or self.video_encoder.getPresetTag() == "x265_nvenc":
                        command += [
                            "-x265-params",
                                "hdr-opt=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc",
                        ]
                    elif self.video_encoder.getPresetTag() == "prores":
                        command += [
                            "-profile:v",
                            "4",
                            "-vendor",
                            "ap10",
                            "-color_range",
                            "full",
                        ]

                command += [
                    "-pix_fmt",
                    self.pixelFormat,

                ]
            command +=[
                f"{self.outputFile}",
            ]

            if self.overwrite:
                command.append("-y")

            if self.slowmo_mode:
                log("Slowmo mode enabled, will not merge audio or subtitles.")

        else: # Benchmark mode

            command = [
                f"{FFMPEG_PATH}",
                "-hide_banner",
                "-loglevel",
                "error",
                "-stats",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-video_size",
                f"{self.width * self.upscaleTimes}x{self.upscaleTimes * self.height}",
                "-pix_fmt",
                "rgb48le" if self.hdr_mode else "rgb24",
                "-r",
                str(self.outputFPS),
                "-i",
                "-",
                "-benchmark",
                "-f",
                "null",
                "-",
            ]

        
        return command

    def get_num_frames_rendered(self):
        return self.framesRendered

    def put_frame_in_write_queue(self, frame):
        
        self.writeQueue.put(frame)

    def write_out_frames(self):
        log("Rendering")
        self.startTime = time.time()

        exit_code: int = 0
        try:
            while True:
                frame = self.writeQueue.get()
                if frame is None:
                    break
                
                self.writeProcess.stdin.buffer.write(frame)

            self.writeProcess.stdin.close()
            self.writeProcess.wait()
            exit_code = self.writeProcess.returncode

            renderTime = time.time() - self.startTime
            self.merge_subtitles()
            log(f"\nTime to complete render: {round(renderTime, 2)}")
            
        except Exception as e:
            log(str(e))
            self.onErroredExit()

        if exit_code != 0:
            self.onErroredExit()
            return
        
        

    def merge_subtitles(self):
        if self.slowmo_mode:
            log("Slowmo mode enabled, skipping subtitle merge.")
            return

        if not self.outputFile:
            log("No output file specified, skipping subtitle merge.")
            return
        
        if self.benchmark:
            log("Benchmark mode enabled, skipping subtitle merge.")
            return

        temp_output = self.outputFile + "-" + str(os.getpid()) + "-temp.mkv"
        os.rename(self.outputFile, temp_output)

        command = [
            f"{FFMPEG_PATH}",
            "-loglevel",
            "error",
            "-i",
            temp_output,
            "-i",
            self.inputFile,
            "-c",
            "copy",
            "-c:s",
            "copy",
            "-map",
            "0",
            "-map",
            "1:s?",
            self.outputFile,
        ]

        log("Merging subtitles with command: " + " ".join(command))

        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                log("Failed to merge subtitles. FFmpeg error:")
                log(result.stderr.decode())
                os.remove(self.outputFile) # Remove incomplete output file
                os.rename(temp_output, self.outputFile)  # Restore original file
                return
            os.remove(temp_output)
            log("Subtitles merged successfully.")
        except Exception as e:
            log("Exception occurred while merging subtitles: " + str(e))
            os.rename(temp_output, self.outputFile)  # Restore original file


    def onErroredExit(self):
        log("FFmpeg failed to render the video.")
        try:
            with open(FFMPEG_LOG_FILE, "r") as f:
                log("FULL FFMPEG LOG:")
                for line in f.readlines():
                    log(line)

            with open(FFMPEG_LOG_FILE, "r") as f:
                for line in f.readlines():
                    if f"[{self.outputFileExtension}" in line:
                        log(line)

            if self.video_encoder.getPresetTag() == "x264_vulkan":
                log("Vulkan encode failed, try restarting the render.")
                log(
                    "Make sure you have the latest drivers installed and your GPU supports vulkan encoding."
                )
        except Exception as e:
            print("Failed to read FFmpeg log file.")
            print(str(e))

        time.sleep(1)
        os._exit(1)

    def __del__(self):
        self.ffmpeg_log.close()

class MPVOutput:
    def __init__(self, FFMpegWrite: FFmpegWrite, width, height,fps, outputFrameChunkSize):
        self.proc = None
        self.startTime = time.time()
        self.FFMPegWrite = FFMpegWrite
        self.outputFrameChunkSize = outputFrameChunkSize
        self.width = width
        self.height = height
        self.fps = fps


    def command(self):
        command = [
        "mpv",
        f"--audio-file={self.FFMPegWrite.inputFile}",
        "--no-config",
        "--cache=yes",
        "--cache-secs=5",                    # Cache 30 seconds of video
        "--demuxer-max-bytes=500Mib",         # Increase max bytes
        "--demuxer-readahead-secs=5",        # Read ahead 30 seconds
        "--demuxer-seekable-cache=yes",       # Enable seekable cache
        "--stream-buffer-size=500MiB",        # Increase buffer size
        "--hr-seek-framedrop=no",            # Prevent frame dropping during seeks
        "-"
        ]
        return command

    def write_out_frames(self):
        with open('mpv_log.txt', "w") as f:
            while not self.FFMPegWrite.writeProcess:
                time.sleep(1)
            self.proc = subprocess_popen_without_terminal(
                self.command(),
                stdin=self.FFMPegWrite.writeProcess.stdout,
                stderr=f,
                stdout=f,
            )
            self.FFMPegWrite.writeProcess.stdout.close()
            self.proc.wait()
            self.stop()
            os._exit(0) # force exit

    def stop(self):
        """
        Stop mpv by closing stdin.
        """
        if self.proc:
            self.proc.terminate()
            self.proc.wait()
