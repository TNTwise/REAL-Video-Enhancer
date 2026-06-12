import logging
import re
import subprocess
from abc import ABC, abstractmethod
from datetime import time

from src.constants import FFMPEG_PATH
import cv2
from src.schemas.request.video_info import InputVideoInfoClientInput

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
    "ictcp",
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
    "bt2020",
    "smpte428",
    "smpte431",
    "smpte432",
    "jedec-p22",
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
    "arib-std-b67",
]

if not __name__ == "__main__":
    from ..utils.LogConfig import get_logger
    from ..utils.Util import subprocess_popen_without_terminal

    logger = get_logger(__name__)

else:
    from Util import subprocess_popen_without_terminal

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class FFMpegInfoWrapper:
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

        self.ffmpeg_output_raw: str = subprocess_popen_without_terminal(
            command, stderr=subprocess.PIPE, errors="replace"
        ).stderr.read()
        self.ffmpeg_output_stripped = self.ffmpeg_output_raw.lower().strip()

        for line in self.ffmpeg_output_raw.split("\n"):
            if "Stream #" in line and "Video" in line:
                self.stream_line = line
                self.ffmpeg_output_raw.replace(line, "")
                break

        for line in self.ffmpeg_output_raw.split("\n"):
            if "Stream #" in line and "Video" in line:
                self.stream_line_2 = line
                self.ffmpeg_output_raw = self.ffmpeg_output_raw.replace(line, "")
                break

        if self.stream_line is None:
            logger.error("No video stream found in the input file.")
            exit(1)

    def get_duration_seconds(self) -> float:
        total_duration: float = 0.0

        duration = re.search(r"duration: (.*?),", self.ffmpeg_output_stripped).groups()[
            0
        ]
        hours, minutes, seconds = duration.split(":")
        total_duration += int(int(hours) * 3600)
        total_duration += int(int(minutes) * 60)
        total_duration += float(seconds)
        return round(total_duration, 2)

    def get_total_frames(self) -> int:
        return int(self.get_duration_seconds() * self.get_fps())

    def get_width_x_height(self) -> list[int]:
        width, height = re.search(
            r"video:.* (\d+)x(\d+)", self.ffmpeg_output_stripped
        ).groups()[:2]

        width = int(width)
        height = int(height)

        rot = abs(self.get_rotation())
        if rot in [90, 270]:
            return [height, width]

        return [width, height]

    def get_rotation(self) -> float:
        m = re.search(
            r"rotate\s*:\s*(-?\d+)", self.ffmpeg_output_stripped, re.IGNORECASE
        )
        if m:
            return float(m.group(1))
        m = re.search(
            r"rotation of (-?\d+\.?\d*) degrees",
            self.ffmpeg_output_stripped,
            re.IGNORECASE,
        )
        if m:
            return float(m.group(1))
        return 0.0

    def get_fps(self) -> float:
        fps = re.search(r"(\d+\.?\d*) fps", self.ffmpeg_output_stripped).groups()[0]
        return float(fps)

    def check_color_opt(self, color_opt: str) -> str | None:
        if self.stream_line:
            if "ffv1" in self.get_codec():
                string_pattern = "1,"
            else:
                string_pattern = "),"
            match color_opt:
                case "Space":
                    color_opt_detected = (
                        self.stream_line_2.split(",")[1].split("(")[1].strip()
                    )
                    if color_opt_detected not in FFMPEG_COLORSPACES:
                        color_opt_detected = (
                            self.stream_line.split(string_pattern)[1]
                            .split(",")[1]
                            .split("/")[0]
                            .strip()
                        )
                        if color_opt_detected not in FFMPEG_COLORSPACES:
                            return None

                case "Primaries":
                    color_opt_detected = (
                        self.stream_line.split(string_pattern)[1].split("/")[1].strip()
                    )
                    if color_opt_detected not in FFMPEG_COLOR_PRIMARIES:
                        return None
                case "Transfer":
                    color_opt_detected = (
                        self.stream_line.split(string_pattern)[1]
                        .split("/")[2]
                        .replace(")", "")
                        .split(",")[0]
                        .strip()
                    )
                    if color_opt_detected not in FFMPEG_COLOR_TRC:
                        return None

            if "progressive" in color_opt_detected.lower():
                return None
            if "unknown" in color_opt_detected.lower():
                return None

            if len(color_opt_detected.strip()) > 1:
                return color_opt_detected

        return None

    def get_color_space(self) -> str:
        try:
            return self.check_color_opt("Space")
        except Exception:
            logger.exception("Can't detect color space.")

    def get_color_primaries(self) -> str:
        try:
            return self.check_color_opt("Primaries")
        except Exception:
            logger.exception("Can't detect color primaries.")

    def get_color_transfer(self) -> str:
        try:
            return self.check_color_opt("Transfer")
        except Exception:
            logger.exception("Can't detect color transfer.")

    def get_pixel_format(self) -> str:
        if self.stream_line:
            try:
                pixel_format = self.stream_line.split(",")[1].split("(")[0].strip()
                return pixel_format
            except Exception:
                logger.exception("Can't detect pixel format.")
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


class OpenCVInfo:
    def __init__(self, video_info_client: InputVideoInfoClientInput):
        logger.info("Getting Input Video Properties")
        self.cap = cv2.VideoCapture(video_info_client.input_file)
        self.ffmpeg_info = FFMpegInfoWrapper(
            video_info_client.input_file, ffmpeg_path=FFMPEG_PATH
        )

    @property
    def is_valid_video(self):
        return self.cap.isOpened() and self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def get_duration_seconds(self) -> float:
        duration = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.input_fps

        return duration

    def get_total_frames(self) -> int:
        fc = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return fc

    @property
    def input_width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def input_height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def rotation(self) -> float:
        return self.ffmpeg_info.get_rotation()

    @property
    def input_fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def color_space(self) -> str:
        return self.ffmpeg_info.get_color_space()

    @property
    def color_transfer(self) -> str:
        return self.ffmpeg_info.get_color_transfer()

    @property
    def color_primaries(self) -> str:
        return self.ffmpeg_info.get_color_primaries()

    @property
    def pixel_format(self) -> str:
        return self.ffmpeg_info.get_pixel_format()

    @property
    def bitrate(self) -> int:
        return self.ffmpeg_info.get_bitrate()

    @property
    def codec(self) -> str:
        return self.ffmpeg_info.get_codec()

    @property
    def bit_depth(self) -> int:
        return self.ffmpeg_info.get_bit_depth()

    @property
    def is_hdr(self) -> bool:
        return self.ffmpeg_info.is_hdr()

    def __del__(self):
        self.cap.release()


__all__ = ["FFMpegInfoWrapper", "OpenCVInfo", "print_video_info"]
