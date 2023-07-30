
import sys
import os
import subprocess
import re
from PyQt5.QtWidgets import QApplication, QMainWindow
import src.getLinkVideo.get_vid_from_link as getLinkedVideo
from src.workers import *
import cv2
from src.return_data import *
import requests
thisdir = os.getcwd()
class GetLinkedWindow(QMainWindow):
    

    def __init__(self,selfdata):
        super(GetLinkedWindow, self).__init__()
        self.ui = getLinkedVideo.Ui_MainWindow()
        self.ui.setupUi(self)
        self.selfdata = self
        self.ui.next.clicked.connect(self.next)
        self.ui.qualityCombo.hide()
        self.ui.qualityLabel.hide()
        self.main = selfdata
        self.setMinimumSize(700, 550)
        self.ui.error_label.setStyleSheet('QLabel#error_label {color: red}')
        
        self.show()
    def get_fps_from_video_link(self,video_link):
        cap = cv2.VideoCapture(video_link)
        if not cap.isOpened():
            self.ui.error_label.setText("Failed to open the video.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps

    
    def next(self):
        self.ui.error_label.clear()
        self.ui.next.hide()
        if 'youtu.be'  in self.ui.plainTextEdit.toPlainText() or 'youtube.com' in self.ui.plainTextEdit.toPlainText():
            #gets data from youtube using youtubedlp
            self.run_ytdl_thread()
            self.main.youtubeFile = True
        else:
            #get data from link here
            
            self.main.youtubeFile = False
            self.main.input_file = self.ui.plainTextEdit.toPlainText()
            self.main.videoName = self.main.input_file.split('/')[-1]
            try:
                response = requests.head(self.main.input_file)
                response.raise_for_status()
                content_type = response.headers.get('Content-Type')
                if content_type and content_type.startswith('video/'):
                    pass
                else:
                    self.ui.error_label.setText(f"Invalid video format. Content-Type: {content_type}")
                    self.ui.next.show()
                    return False
            except requests.exceptions.RequestException as e:
                if response.status_code != 403: # ignores 403 errors cause they mostly work even when the exception is raised :\
                    self.ui.error_label.setText(f'{e}')
                    self.ui.next.show()
                    return False
            self.main.fps=int(VideoName.return_video_framerate(self.main.input_file))
            self.main.localFile=False
            self.main.showChangeInFPS(False)
            self.main.fc = VideoName.return_video_frame_count(self.main.input_file)
            self.ytVidRes = self.ui.qualityCombo.currentText()
            self.main.download_youtube_video_command = self.main.input_file
            self.main.input_file = f'{thisdir}/{self.main.videoName}'
            window.close()
            
    def run_ytdl_thread(self):
        self.thread = QThread()
        self.worker = downloadVideo(self,self.ui.plainTextEdit.toPlainText())
        self.ui.plainTextEdit.setDisabled(True)
        self.ui.next.hide()
        self.ui.qualityCombo.clear()
        

        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.report_progress)
        
        
        # Step 6: Start the thread
        
        self.thread.start()
        self.worker.addRes.connect(self.addRes)
        # Final resets
        
        self.worker.finished.connect(
            self.end_DownloadofData
        )    
    def addRes(self,res):
        self.ui.qualityCombo.addItem(res)
    def report_progress(self,result):
        if result == f"""ERROR: [generic] None: '{self.ui.plainTextEdit.toPlainText()}' is not a valid URL. Set --default-search "ytsearch" (or run  yt-dlp "ytsearch:{self.ui.plainTextEdit.toPlainText()}" ) to search YouTube\n""": 
            self.ui.error_label.setText("Invalid URL")
        else:
            self.ui.error_label.setText(result)
        self.ui.qualityCombo.hide()
        self.ui.qualityLabel.hide()
    def get_youtube_video_duration(self,url):
        try:
            result = subprocess.run([f'{thisdir}/bin/yt-dlp_linux', self.ui.plainTextEdit.toPlainText(), '--get-duration'], capture_output=True, text=True)
            if result.returncode == 0:
                duration_str = result.stdout.strip()
                duration_parts = duration_str.split(':')
                if len(duration_parts) == 3:
                    # Convert HH:MM:SS to seconds
                    duration_seconds = int(duration_parts[0]) * 3600 + int(duration_parts[1]) * 60 + int(duration_parts[2])
                    return duration_seconds
                if len(duration_parts) == 2:
                     return int(duration_parts[0]) * 60 + int(duration_parts[1])
                if len(duration_parts) == 1:
                     return int(duration_parts[0])
            return None
        except Exception as e:
            print("Error:", e)
            return None
        
    def get_youtube_video_name(self,url):
        try:
            result = subprocess.run([f'{thisdir}/bin/yt-dlp_linux', '--get-title', url], capture_output=True, text=True)
            if result.returncode == 0:
                video_name = result.stdout.strip()
                return video_name
            return None
        except Exception as e:
            print("Error:", e)
            return None
    def end_DownloadofData(self,dict_res_id_fps):
        global return_command
        
        
        self.ui.next.clicked.disconnect(self.next)
        self.ui.next.clicked.connect(self.gen_youtubedlp_command)
        self.ui.next.show()
        self.ui.qualityLabel.show()
        self.ui.qualityCombo.show()
        
        self.dict_res_id_fps = dict_res_id_fps
    def gen_youtubedlp_command(self):
        
        self.main.download_youtube_video_command = (f'{thisdir}/bin/yt-dlp_linux -f {self.dict_res_id_fps[self.ui.qualityCombo.currentText()][0]} "{self.ui.plainTextEdit.toPlainText()}" -o "{self.main.input_file}" && {thisdir}/bin/yt-dlp_linux -f 140 "{self.ui.plainTextEdit.toPlainText()}"  -o {thisdir}/audio.m4a')
        self.main.fps=int(self.dict_res_id_fps[self.ui.qualityCombo.currentText()][1])
        self.main.localFile=False
        self.main.showChangeInFPS(False)
        self.main.fc = int(self.main.fps*self.duration)
        self.ytVidRes = self.ui.qualityCombo.currentText()
        self.main.ytVidWidth = self.ui.qualityCombo.currentText().split('x')[0]
        self.main.ytVidHeight = self.ui.qualityCombo.currentText().split('x')[1]
        window.close()
def get_linked_video(self):
    global window
    window = GetLinkedWindow(self)
    window.show()
    

#./yt-dlp_linux -f best "https://www.youtube.com/watch?v=k9hv2l3NOZU"  -o - | ffmpeg -i - frames/%08d.jpg && ./yt-dlp_linux -f 140 "https://www.youtube.com/watch?v=k9hv2l3NOZU"  -o - | ffmpeg -i -  output.m4a