import cv2
import os
import warnings
import sys
import requests
import stat

cwd = os.getcwd()

with open(os.path.join(cwd, "log.txt"), "w") as f:
    pass


def getPlatform() -> str:
    """
    Returns the current OS that the app is running on
    Windows: win32
    MacOS: darwin
    Linux: linux
    """
    return sys.platform


def pythonPath() -> str:
    return (
        os.path.join(cwd, "python", "bin", "python3")
        if getPlatform() == "darwin" or getPlatform() == "linux"
        else os.path.join(cwd, "python", "bin", "python3.exe")
    )


def move(prev: str, new: str):
    """
    moves a file from prev to new
    """
    os.rename(prev, new)


def ffmpegPath() -> str:
    return (
        os.path.join(cwd, "ffmpeg", "ffmpeg")
        if getPlatform() == "darwin" or getPlatform() == "linux"
        else os.path.join(cwd, "ffmpeg", "ffmpeg.exe")
    )


def makeExecutable(file_path):
    st = os.stat(file_path)
    os.chmod(file_path, st.st_mode | stat.S_IEXEC)


def warnAndLog(message: str):
    warnings.warn(message)
    log("WARN: " + message)


def createDirectory(dir: str):
    if not os.path.exists(dir):
        os.mkdir(dir)


def printAndLog(message: str, separate=False):
    """
    Prints and logs a message to the log file
    separate, if True, activates the divider
    """
    if separate:
        message = message + "\n" + "---------------------"
    print(message)
    log(message=message)


def log(message: str):
    with open(os.path.join(cwd, "log.txt"), "a") as f:
        f.write(message + "\n")


def currentDirectory():
    return cwd


def removeFile(file):
    os.remove(file)


def checkIfDeps() -> bool:
    """
    Checks if python or ffmpeg is installed, and if not returns false.
    """
    if os.path.isfile(ffmpegPath()) == False or os.path.isfile(pythonPath()) == False:
        return False
    return True


def downloadFile(link, downloadLocation):
    response = requests.get(
        link,
        stream=True,
    )

    with open(downloadLocation, "wb") as f:
        for chunk in response.iter_content(chunk_size=128):
            f.write(chunk)


def checkValidVideo(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Couldn't open the video file '{video_path}'")
        return False

    ret, frame = cap.read()
    if not ret:
        print(f"Error: Couldn't read frames from the video file '{video_path}'")
        return False

    cap.release()

    return True


def getVideoRes(video_path) -> list[int, int]:
    """
    Takes in a video path
    Uses opencv to detect the resolution of the video
    returns [width,height]
    """
    cap = cv2.VideoCapture(video_path)

    # Get the resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution = [width, height]

    cap.release()

    return resolution


def getVideoFPS(video_path) -> float:
    """
    Takes in a video path
    Uses opencv to detect the FPS of the video
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    return fps


def getDefaultOutputVideo(outputPath):
    pass


def getVideoLength(video_path) -> int:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Error: Could not open video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    duration = total_frames / fps

    cap.release()

    return duration


def getVideoFrameCount(video_path) -> int:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Error: Could not open video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return total_frames
