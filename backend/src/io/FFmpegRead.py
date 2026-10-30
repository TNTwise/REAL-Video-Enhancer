import queue
import subprocess
import cv2
import numpy as np

from .ReadBuffer import ReadBuffer
from ..utils.Util import (
    log,
    subprocess_popen_without_terminal,
)
from ..utils.Frame import Frame

class FFmpegRead(ReadBuffer):
    def __init__(
            self, 
            inputFile, 
            width, 
            height, 
            start_time, 
            end_time, 
            borderX, 
            borderY, 
            hdr_mode,
            backend: str = "pytorch",
            device: str = "cuda",
            gpu_id: int = 0,
            dtype: str = "float16",
            color_space=None, 
            color_primaries=None, 
            color_transfer=None, 
            input_pixel_format: str | None = None,
            ffmpeg_path: str = "./bin/ffmpeg",
        ):
        
        self.inputFile = inputFile
        self.width = width
        self.height = height
        self.start_time = start_time
        self.end_time = end_time
        self.borderX = borderX
        self.borderY = borderY
        self.hdr_mode = hdr_mode
        self.backend = backend
        self.device = device
        self.gpu_id = gpu_id
        self.dtype = dtype
        self.color_space = color_space
        self.color_primaries = color_primaries
        self.color_transfer = color_transfer
        self.input_pixel_format = input_pixel_format
        self.yuv420pMOD = self.input_pixel_format == "yuv420p" and not self.hdr_mode
        self.ffmpeg_path = ffmpeg_path
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
            f"{self.ffmpeg_path}",
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
            "-",
            
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
            frame = Frame(self.backend, self.width, self.height, self.device, self.gpu_id, self.hdr_mode, self.dtype)
            frame.set_frame_bytes(chunk)
            self.readQueue.put(frame)
        self.readQueue.put(None)

    def get(self) -> Frame:
        return self.readQueue.get()

    def close(self):
        self.readProcess.stdout.close()
        self.readProcess.terminate()
