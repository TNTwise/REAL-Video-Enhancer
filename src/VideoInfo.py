import subprocess
import re
import os
from .constants import FFMPEG_PATH


class RVEBackendWrapper:
    def __init__(self, input_file: str):
        self.input_file = input_file
        self._get_ffmpeg_info()

    def _get_ffmpeg_info(self):
        from .constants import BACKEND_PATH, PYTHON_EXECUTABLE_PATH
        from .Util import log, subprocess_popen_without_terminal

        command = [
            PYTHON_EXECUTABLE_PATH,
            os.path.join(BACKEND_PATH, "rve-backend.py"),
            "--print_video_info",
            self.input_file,
            "--ffmpeg_path",
            f"{FFMPEG_PATH}",
        ]

        result = subprocess_popen_without_terminal(
            command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, errors="replace"
        )
        stderr_output = result.stderr.read().strip()
        stdout_output = result.stdout.read().strip()

        # Try both stdout and stderr
        self.rve_backend_output = stdout_output if stdout_output else stderr_output
        log(f"RVE Backend Output:\n{self.rve_backend_output}")

    def _extract_value(self, label: str) -> str:
        """Extract value from backend output for a given label"""
        # Debug: show what we're looking for

        # Try multiple patterns to be more flexible
        # Using a variable for backslash gets around f-string error on line 39
        backslash = "\\"
        patterns = [
            rf"{label}:\s*(.+)",
            rf"{label.lower()}:\s*(.+)",
            rf"{label.replace(' ', backslash + 's+')}:\s*(.+)",
        ]

        for i, pattern in enumerate(patterns):
            match = re.search(
                pattern, self.rve_backend_output, re.IGNORECASE | re.MULTILINE
            )
            if match:
                return match.group(1).strip()

        return ""

    def get_duration_seconds(self) -> float:
        """Get video duration in seconds"""
        duration_str = self._extract_value("Duration")
        if duration_str:
            return float(duration_str.replace(" seconds", ""))
        return 0.0

    def get_total_frames(self) -> int:
        """Get total number of frames"""
        frames_str = self._extract_value("Total Frames")
        if frames_str:
            return int(frames_str)
        return 0

    def get_width_x_height(self) -> tuple[int, int]:
        """Get video resolution as (width, height)"""
        resolution_str = self._extract_value("Resolution")
        if resolution_str and "x" in resolution_str:
            width, height = resolution_str.split("x")
            return int(width.strip()), int(height.strip())
        return 0, 0

    def get_fps(self) -> float:
        """Get frames per second"""
        fps_str = self._extract_value("FPS")
        if fps_str:
            return float(fps_str)
        return 0.0

    def get_color_space(self) -> str:
        """Get color space"""
        color_space = self._extract_value("Color Space")
        return color_space if not "None" in color_space else None

    def get_color_transfer(self) -> str:
        """Get color transfer"""
        color_transfer = self._extract_value("Color Transfer")
        return color_transfer if not "None" in color_transfer else None

    def get_color_primaries(self) -> str:
        """Get color primaries"""
        color_primaries = self._extract_value("Color Primaries")
        return color_primaries if not "None" in color_primaries else None

    def get_pixel_format(self) -> str:
        """Get pixel format"""
        return self._extract_value("Pixel Format")

    def get_codec(self) -> str:
        """Get video codec"""
        return self._extract_value("Video Codec")

    def get_bitrate(self) -> str:
        """Get video bitrate in kbps"""
        bitrate_str = self._extract_value("Video Bitrate")
        if bitrate_str:
            return bitrate_str
        return 0

    def is_hdr(self) -> bool:
        """Check if video is HDR"""
        hdr_str = self._extract_value("Is HDR")
        return hdr_str.lower() == "true"

    def get_bit_depth(self) -> int:
        """Get bit depth"""
        bit_depth_str = self._extract_value("Bit Depth")
        if bit_depth_str:
            return int(bit_depth_str.replace(" bit", ""))
        return 0


class VideoLoader:
    def __init__(self, inputFile):
        self.inputFile = inputFile

    def loadVideo(self):
        self.ffmpeg_info = RVEBackendWrapper(self.inputFile)

    def isValidVideo(self):
        try:
            disabled_extensions = ["txt", "jpg", "jpeg", "png", "bmp", "webp"]
            file_extension = self.inputFile.split(".")[-1].lower()
            return (
                self.ffmpeg_info.get_total_frames() > 1
                and file_extension not in disabled_extensions
            )
        except Exception as e:
            return False

    def getData(self):
        self.width, self.height = self.ffmpeg_info.get_width_x_height()

        self.bitrate = self.ffmpeg_info.get_bitrate()
        self.videoContainer = self.inputFile.split(".")[-1]
        self.codec_str = self.ffmpeg_info.get_codec()

        self.fps = self.ffmpeg_info.get_fps()
        self.total_frames = int(self.ffmpeg_info.get_total_frames())
        self.duration = self.total_frames / self.fps
        self.color_space = self.ffmpeg_info.get_color_space()
        self.color_transfer = self.ffmpeg_info.get_color_transfer()
        self.color_primaries = self.ffmpeg_info.get_color_primaries()
        self.pixel_format = self.ffmpeg_info.get_pixel_format()
        self.is_hdr = self.ffmpeg_info.is_hdr()
        self.bit_depth = self.ffmpeg_info.get_bit_depth()
