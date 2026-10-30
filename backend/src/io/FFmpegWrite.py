import os
import subprocess
import time
import queue

from .WriteBuffer import WriteBuffer
from ..utils.Util import (
    log,
    subprocess_popen_without_terminal,
)
from ..utils.Encoders import EncoderSettings

class FFmpegWrite(WriteBuffer):
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
        ffmpeg_path: str = "./bin/ffmpeg",
        ffmpeg_log_file: str = "ffmpeg_log.txt",
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
        self.merge_subtitles = merge_subtitles
        self.writeQueue = queue.Queue(maxsize=25)
        self.previewFrame = None
        self.framesRendered: int = 1
        self.writeProcess = None
        self.color_space = color_space
        self.color_primaries = color_primaries
        self.color_transfer = color_transfer
        self.ffmpeg_path = ffmpeg_path
        self.ffmpeg_log_file = ffmpeg_log_file
        self.outputFPS = (
            (self.fps * self.interpolateFactor)
            if not self.slowmo_mode
            else self.fps
        )
        self.ffmpeg_log = open(self.ffmpeg_log_file, "w", encoding='utf-8')
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
                f"{self.ffmpeg_path}",
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
                f"{self.ffmpeg_path}",
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
                    # Input 1: original file for audio/subtitles.
                    # Put timestamp hygiene flags *before* the input they apply to.
                    "-fflags",
                    "+genpts",
                    "-i",
                    f"{self.inputFile}",
                    "-map",
                    "0:v",  # Map video stream from input 0
                    "-map",
                    "1:a?",
                    "-map",
                    "1:s?",
                ]

                # Output timestamp/interleave hygiene.
                command += [
                    "-avoid_negative_ts",
                    "make_zero",
                    "-max_interleave_delta",
                    "0",
                    "-muxpreload",
                    "0",
                    "-muxdelay",
                    "0",
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

                # MP4/MOV: improve seekability by moving the moov atom to the front.
                if self.outputFile and self.outputFileExtension.lower() in ("mp4", "mov", "m4v"):
                    command += [
                        "-movflags",
                        "+faststart",
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
                f"{self.ffmpeg_path}",
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
            log(f"\nTime to complete render: {round(renderTime, 2)}")
            
        except Exception as e:
            log(str(e))
            self.onErroredExit()

        if exit_code != 0:
            self.onErroredExit()
            return
        
        

    def onErroredExit(self):
        log("FFmpeg failed to render the video.")
        try:
            with open(self.ffmpeg_log_file, "r") as f:
                log("FULL FFMPEG LOG:")
                for line in f.readlines():
                    log(line)

            with open(self.ffmpeg_log_file, "r") as f:
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
