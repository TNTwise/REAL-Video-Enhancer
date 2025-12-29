from abc import ABC, abstractmethod
from typing import List
import subprocess
import re
import cv2
from typing import Optional
import sys


FFMPEG_COLORSPACES = [
    "rgb",
    "bt709",
    "unknown",
    "reserved",
    "fcc",
    "bt470bg",
    "smpte170m",
    "smpte240m",
    "ycgco",
    "bt2020nc",
    "bt2020c",
    "smpte2085",
    "chroma-derived-nc",
    "chroma-derived-c",
    "ictcp"
]

FFMPEG_COLOR_PRIMARIES = [
    "reserved0",
    "bt709",
    "unknown",
    "reserved",
    "bt470m",
    "bt470bg",
    "smpte170m",
    "smpte240m",
    "film",
    "bt2020",
    "smpte428",
    "smpte431",
    "smpte432",
    "jedec-p22"
]
FFMPEG_COLOR_TRC = [
    "reserved0",
    "bt709",
    "unknown",
    "reserved",
    "bt470m",
    "bt470bg",
    "smpte170m",
    "smpte240m",
    "linear",
    "log100",
    "log316",
    "iec61966-2-4",
    "bt1361e",
    "iec61966-2-1",
    "bt2020-10",
    "bt2020-12",
    "smpte2084",
    "smpte428",
    "arib-std-b67"
]

if not __name__ == "__main__":
    from .Util import log, subprocess_popen_without_terminal

else:
    from Util import log, subprocess_popen_without_terminal

class VideoInfo(ABC):
    @abstractmethod
    def get_duration_seconds(self) -> float: ...
    @abstractmethod
    def get_total_frames(self) -> int: ...
    @abstractmethod
    def get_width_x_height(self) -> List[int]: ...
    @abstractmethod
    def get_fps(self) -> float: ...
    @abstractmethod
    def get_color_space(self) -> str: ...
    @abstractmethod
    def get_pixel_format(self) -> str: ...
    @abstractmethod
    def get_color_transfer(self) -> str: ...
    @abstractmethod
    def get_color_primaries(self) -> str: ...
    @abstractmethod
    def get_bitrate(self) -> int: ...
    @abstractmethod
    def get_codec(self) -> str: ...
    @abstractmethod
    def is_hdr(self) -> bool: ...
    @abstractmethod
    def get_bit_depth(self) -> int: ...

class FFMpegInfoWrapper(VideoInfo):
    def __init__(self, input_file: str, ffmpeg_path: str = "./bin/ffmpeg"):
        self.input_file = input_file
        self.ffmpeg_path = ffmpeg_path
        self.stream_line = None
        self.stream_line_2 = None
        self._get_ffmpeg_info()

    def _get_ffmpeg_info(self):
        command = [
                self.ffmpeg_path,
                "-i",
                self.input_file,
                "-t",
                "00:00:00",
                "-f",
                "null",
                "/dev/null",
                "-hide_banner",
                
                
        ]

        self.ffmpeg_output_raw:str = subprocess_popen_without_terminal(command,  stderr=subprocess.PIPE, errors="replace").stderr.read()
        self.ffmpeg_output_stripped = self.ffmpeg_output_raw.lower().strip()
        try:
            for line in self.ffmpeg_output_raw.split("\n"):
                if "Stream #" in line and "Video" in line:
                    self.stream_line = line
                    self.ffmpeg_output_raw = self.ffmpeg_output_raw.replace(line, "")
                    break

            for line in self.ffmpeg_output_raw.split("\n"):
                if "Stream #" in line and "Video" in line:
                    self.stream_line_2 = line
                    self.ffmpeg_output_raw = self.ffmpeg_output_raw.replace(line, "")
                    break
            if self.stream_line is None:
                log("No video stream found in the input file.")
        except Exception:
            log(f"ERROR: Input file seems to have no video stream!", file=sys.stderr)
            exit(1)
            

    def get_duration_seconds(self) -> float:
        total_duration:float = 0.0

        duration = re.search(r"duration: (.*?),", self.ffmpeg_output_stripped).groups()[0]
        hours, minutes, seconds = duration.split(":")
        total_duration += int(int(hours) * 3600)
        total_duration += int(int(minutes) * 60)
        total_duration += float(seconds)
        return round(total_duration, 2)

    def get_total_frames(self) -> int:
        return int(self.get_duration_seconds() * self.get_fps())

    def get_width_x_height(self) -> List[int]:
        width, height = re.search(r"video:.* (\d+)x(\d+)",self.ffmpeg_output_stripped).groups()[:2]
        return [int(width), int(height)]

    def get_fps(self) -> float:
        fps = re.search(r"(\d+\.?\d*) fps", self.ffmpeg_output_stripped).groups()[0]
        return float(fps)
    
    def check_color_opt(self, color_opt:str) -> str | None:
        if self.stream_line:
            if "ffv1" in self.get_codec():
                string_pattern = "1,"
            else:
                string_pattern = "),"
            try:
                match color_opt:
                    case "Space":
                        color_opt_detected = self.stream_line_2.split(",")[1].split("(")[1].strip()
                        if color_opt_detected not in FFMPEG_COLORSPACES:
                            color_opt_detected = self.stream_line.split(string_pattern)[1].split(",")[1].split("/")[0].strip()
                            if color_opt_detected not in FFMPEG_COLORSPACES:
                                return None

                    case "Primaries":
                        color_opt_detected = self.stream_line.split(string_pattern)[1].split("/")[1].strip()
                        if color_opt_detected not in FFMPEG_COLOR_PRIMARIES:
                            return None
                    case "Transfer":
                        color_opt_detected = self.stream_line.split(string_pattern)[1].split("/")[2].replace(")","").split(",")[0].strip()
                        if color_opt_detected not in FFMPEG_COLOR_TRC:
                            return None

                if "progressive" in color_opt_detected.lower():
                    return None
                if "unknown" in color_opt_detected.lower():
                    return None
                
                if len(color_opt_detected.strip()) > 1:
                    return color_opt_detected
                
            except Exception:
                return None
        return None
    
    def get_color_space(self) -> str:
        return self.check_color_opt("Space")

    def get_color_primaries(self) -> str:
        return self.check_color_opt("Primaries")

    def get_color_transfer(self) -> str:
        return self.check_color_opt("Transfer")

    def get_pixel_format(self) -> str:
        if self.stream_line:
            try:
                pixel_format = self.stream_line.split(",")[1].split("(")[0].strip()
                return pixel_format
            except Exception:
                log("ERROR: Cant detect pixel format.")
        return None
    
    def is_hdr(self) -> bool:
        hdr_indicators = ["bt2020", "pq", "hdr10", "dolby vision", "hlg"]
        for indicator in hdr_indicators:
            if indicator in self.ffmpeg_output_stripped:
                return True
        return False
    
    def get_bitrate(self) -> int:
        bitrate = re.search(r"bitrate: (\d+)", self.ffmpeg_output_stripped)
        if bitrate:
            return int(bitrate.groups()[0])
        return 0
    
    def get_codec(self) -> str:
        codec = re.search(r"video: (\w+)", self.ffmpeg_output_stripped)
        if codec:
            return codec.groups()[0]
        return "unknown"
    
    def get_bit_depth(self) -> int:
        return 10 if "p10le" in self.ffmpeg_output_stripped else 8
    


class OpenCVInfo(VideoInfo):
    def __init__(self, input_file: str, start_time: Optional[float] = None, end_time: Optional[float] = None, ffmpeg_path: str = "./bin/ffmpeg"):
        log("Getting Input Video Properties")
        self.input_file = input_file
        self.start_time = start_time
        self.end_time = end_time
        self.cap = cv2.VideoCapture(input_file)
        self.ffmpeg_info = FFMpegInfoWrapper(input_file, ffmpeg_path=ffmpeg_path)

    def is_valid_video(self):
        #frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        #log(f"Frame count: {frame_count}")
        #if frame_count <= 1:
        #    log("Invalid video: Frame count is less than or equal to 1.")
        #    return False
        
        return self.cap.isOpened() and self.cap.get(cv2.CAP_PROP_FRAME_COUNT) 

    def get_duration_seconds(self) -> float:
        duration = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.get_fps()

        if self.start_time is not None and self.end_time is not None:
            duration = self.end_time - self.start_time
        elif self.start_time and not self.end_time:
            duration = duration - self.start_time
        elif self.end_time and not self.start_time:
            duration = self.end_time
        return duration

    def get_total_frames(self) -> int:
        
        if self.start_time or self.end_time:
            fc = int(self.get_duration_seconds() * self.get_fps())
        else:
            fc =  int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return fc

    def get_width_x_height(self) -> List[int]:
        res = [int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        return res


    def get_fps(self) -> float:
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps
    
    def get_color_space(self) -> str:
        return self.ffmpeg_info.get_color_space()
    
    def get_pixel_format(self) -> str:
        return self.ffmpeg_info.get_pixel_format()
    
    def get_color_transfer(self) -> str:
        return self.ffmpeg_info.get_color_transfer()

    def get_color_primaries(self) -> str:
        return self.ffmpeg_info.get_color_primaries()

    def get_bitrate(self) -> int:
        return self.ffmpeg_info.get_bitrate()
    
    def get_codec(self) -> str:
        return self.ffmpeg_info.get_codec()
    
    def is_hdr(self) -> bool:
        return self.ffmpeg_info.is_hdr()
    
    def get_bit_depth(self) -> int:
        return self.ffmpeg_info.get_bit_depth()
    

    def __del__(self):
        self.cap.release()

def print_video_info(video_info: VideoInfo):
    print(f"Duration: {video_info.get_duration_seconds()} seconds")
    print(f"Total Frames: {video_info.get_total_frames()}")
    print(f"Resolution: {video_info.get_width_x_height()[0]}x{video_info.get_width_x_height()[1]}")
    print(f"FPS: {video_info.get_fps()}")
    print(f"Color Space: {video_info.get_color_space()}")
    print(f"Color Transfer: {video_info.get_color_transfer()}")
    print(f"Color Primaries: {video_info.get_color_primaries()}")
    print(f"Pixel Format: {video_info.get_pixel_format()}")
    print(f"Video Codec: {video_info.get_codec()}")
    print(f"Video Bitrate: {video_info.get_bitrate()} kbps")
    print(f"Is HDR: {video_info.is_hdr()}")
    print(f"Bit Depth: {video_info.get_bit_depth()}")

__all__ = ["FFMpegInfoWrapper", "OpenCVInfo", "print_video_info"]

if __name__ == "__main__":
    video_path = "/home/pax/Downloads/ffv1_youtube_test2.mkv"
    #video_path = "/home/pax/Documents/test/LG New York HDR UHD 4K Demo.ts"
    #video_path = "/home/pax/Documents/test/out.mkv"
    #video_path = "/home/pax/Videos/TVアニメ「WIND BREAKER Season 2」ノンクレジットオープニング映像「BOYZ」SixTONES [AWlUVr7Du04]_gmfss-pro_deh264-span_janai-v2_72.0fps_3840x2160.mkv"
    """print("Using FFMpeg:")
    video_info = FFMpegInfoWrapper(video_path)
    print(f"Duration: {video_info.get_duration_seconds()} seconds")
    print(f"Total Frames: {video_info.get_total_frames()}")
    print(f"Resolution: {video_info.get_width_x_height()}")
    print(f"FPS: {video_info.get_fps()}")
    print(f"Color Space: {video_info.get_color_space()}")
    print("\nUsing OpenCV:")"""
    video_info = OpenCVInfo(video_path)
    print(f"Duration: {video_info.get_duration_seconds()} seconds")
    print(f"Total Frames: {video_info.get_total_frames()}")
    print(f"Resolution: {video_info.get_width_x_height()}")
    print(f"FPS: {video_info.get_fps()}")
    print(f"Color Space: {video_info.get_color_space()}")
    print(f"Color Transfer: {video_info.get_color_transfer()}")
    print(f"Color Primaries: {video_info.get_color_primaries()}")
    print(f"Pixel Format: {video_info.get_pixel_format()}")
    print(f"Video Codec: {video_info.get_codec()}")
    print(f"Video Bitrate: {video_info.get_bitrate()} kbps")
    print(f"Is HDR: {video_info.is_hdr()}")
    print(f"Bit Depth: {video_info.get_bit_depth()}")