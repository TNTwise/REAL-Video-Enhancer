import yt_dlp
import validators
import os
from src.Util import (
    checkValidVideo,
    getVideoFPS,
    getVideoRes,
    getVideoLength,
    getVideoFrameCount,
    getVideoEncoder,
    getVideoBitrate,
)


class VideoLoader:
    def __init__(self, inputFile):
        self.inputFile = inputFile

    def getDataFromLocalVideo(self):
        if checkValidVideo(self.inputFile):
            self.isVideoLoaded = True
            # gets width and height from the res
            self.videoWidth, self.videoHeight = getVideoRes(self.inputFile)
            # get fps
            self.videoFps = getVideoFPS(self.inputFile)
            # get video length
            self.videoLength = getVideoLength(self.inputFile)
            # get video frame count
            self.videoFrameCount = getVideoFrameCount(self.inputFile)
            # get video encoder
            self.videoEncoder = getVideoEncoder(self.inputFile)
            # get video bitrate
            self.videoBitrate = getVideoBitrate(self.inputFile)
            # get video codec
            self.videoCodec = getVideoEncoder(self.inputFile)
            self.videoContainer = os.path.splitext(self.inputFile)[1]

    def getDataFromYoutubeVideo(self):
        ydl_opts = {"format": "bestvideo+bestaudio/best"}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(self.inputFile, download=False)
        self.videoContainer = info_dict["ext"]
        self.inputFile = info_dict["title"] + self.videoContainer
        self.videoWidth = info_dict["width"]
        self.videoHeight = info_dict["height"]
        self.videoFps = info_dict["fps"]
        self.videoEncoder = info_dict["vcodec"]
        self.videoBitrate = info_dict["vbr"]
        self.videoFrameCount = int(info_dict["duration"] * info_dict["fps"])

    def getData(self):
        return (
            self.videoWidth,
            self.videoHeight,
            self.videoFps,
            self.videoLength,
            self.videoFrameCount,
            self.videoEncoder,
            self.videoBitrate,
            self.videoContainer,
        )


class VideoInputHandler(VideoLoader):
    def __init__(self, inputText):
        self.inputText = inputText
        super().__init__(inputText)

    def isYoutubeLink(self):
        url = self.inputText
        return validators.url(url) and "youtube.com" in url or "youtu.be" in url

    def isValidVideoFile(self):
        return checkValidVideo(self.inputText)
    
    def isValidYoutubeLink(self):
        ydl_opts = {
        'quiet': True,  # Suppress output
        'noplaylist': True,  # Only check single video, not playlists
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                # Extract info about the video
                info_dict = ydl.extract_info(self.inputText, download=False)
                # Check if there are available formats
                if info_dict.get('formats'):
                    return True  # Video is downloadable
                else:
                    return False  # No formats available
            except Exception as e:
                print(f"Error occurred: {e}")
                return False
        
