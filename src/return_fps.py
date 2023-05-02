import cv2
class Fps:
    def return_video_fps(videopath):
        video=cv2.VideoCapture(fr'{videopath}')
        return video.get(cv2.CAP_PROP_FPS)