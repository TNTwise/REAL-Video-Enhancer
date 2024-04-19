import cv2
import os
import src.programData.thisdir
import platform
from src.misc.log import log

thisdir = src.programData.thisdir.thisdir()


class Fps:
    def return_video_fps(videopath):
        """Returns the frames per second (FPS) of the video."""
        video = cv2.VideoCapture(rf"{videopath}")
        return video.get(cv2.CAP_PROP_FPS)


class VideoName:
    def return_video_name(videopath):
        """Returns the name of the video file."""
        return os.path.basename(videopath)

    def return_video_framerate(videopath):
        """Returns the frame rate of the video."""
        video = cv2.VideoCapture(videopath)
        return video.get(cv2.CAP_PROP_FPS)

    def return_video_frame_count(videopath):
        """Returns the total number of frames in the video."""
        video = cv2.VideoCapture(videopath)
        return video.get(cv2.CAP_PROP_FRAME_COUNT)

    def return_video_resolution(videopath):
        """Returns the resolution (width and height) of the video."""
        video = cv2.VideoCapture(videopath)
        width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return [width, height]


class ManageFiles:
    def create_folder(folderpath):
        """Creates a folder if it doesn't exist."""
        if os.path.exists(folderpath) == False:
            os.system(f'mkdir -p "{folderpath}"')

    def create_file(filepath):
        """Creates a file if it doesn't exist."""
        if os.path.isfile(filepath) == False:
            os.system(f'touch "{filepath}"')

    def isfile(filepath):
        """Checks if a file exists."""
        return os.path.isfile(filepath)

    def isfolder(folderpath):
        """Checks if a folder exists."""
        return os.path.exists(folderpath)


def ceildiv(a, b):
    """Returns the ceiling of the division of two numbers."""
    return -(a // -b)


import multiprocessing


def returnCodec(codec):
    """Returns the codec based on the given codec type."""
    if codec == "264":
        return "libx264"
    if codec == "265":
        return "libx265"
    if codec == "VP9":
        return "libvpx-vp9"
    if codec == "ProRes":
        return "prores -profile:v 2"
    if codec == "Lossless":
        return "copy"
    if codec == "AV1":
        if multiprocessing.cpu_count() >= 8:
            cpus = 8
        else:
            cpus = multiprocessing.cpu_count()
        return f"libaom-av1 -tiles 2x2 -b:v 0 -row-mt 1 -cpu-used {cpus}"


def returnCRFFactor(crffactor, encoder):
    """Returns the Constant Rate Factor (CRF) factor based on the encoder and CRF factor."""
    if "av1" in encoder:
        log("av1 crf")
        crf = int(crffactor) + 12

    elif "vp9" in encoder.lower():
        log("vp9 crf")
        crf = int(crffactor) + 12

    elif "265" in encoder.lower():
        log("265 crf")
        crf = int(crffactor) + 5

    elif "prores" in encoder.lower():
        log("prores crf")
        crf = int(crffactor) + 5

    elif "lossless" in encoder.lower():
        return ""
    else:
        log("264 crf")
        crf = crffactor
        "10"
        "14"
        "18"
        "20"
        "22"
    return f"-crf {crf}"


""" 
Codec Options



libvpx-vp9 - address bitrate based on quality setting

prores: 
pair these to quality settings
-profile:
    -1: auto (default)
    0: proxy ≈ 45Mbps YUV 4:2:2
    1: lt ≈ 102Mbps YUV 4:2:2
    2: standard ≈ 147Mbps YUV 4:2:2
    3: hq ≈ 220Mbps YUV 4:2:2
    4: 4444≈ 330Mbps YUVA 4:4:4:4
    5: 4444xq ≈ 500Mbps YUVA 4:4:4:4




"""


def returnContainer(codec):
    """Returns the container format based on the codec."""
    if "264" in codec or codec == "libx264":
        return "mp4"
    if "265" in codec or codec == "libx265":
        return "mp4"
    if "vp9" in codec.lower() or codec == "libvpx-vp9":
        return "mp4"
    if "prores" in codec.lower() or codec == "prores -profile:v 2":
        return "mov"
    if codec == "copy" or codec == "Lossless":
        return "mkv"
    if "av1" in codec.lower() or codec == "libaom-av1":
        return "mkv"
    log("codec:Fallback")
    return "mkv"


def returnOperatingSystem():
    """
    Returns the current operating system that the program is running on.
    Instead of Darwin, it will return MacOS.
    """
    operating_system = platform.system()
    if operating_system == "Darwin":
        return "MacOS"
    return operating_system
