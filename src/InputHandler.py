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
        super().__init__(inputText)

    def afterSelect(self):
        self.outputFileText.setEnabled(True)
        self.outputFileSelectButton.setEnabled(True)
        self.isVideoLoaded = True
        self.updateVideoGUIDetails()
