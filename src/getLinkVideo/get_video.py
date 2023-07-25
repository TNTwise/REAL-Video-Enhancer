
import sys
import os
import subprocess
import re
from PyQt5.QtWidgets import QApplication, QMainWindow
import src.getLinkVideo.get_vid_from_link as getLinkedVideo
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
   
    def next(self):
        
        if 'youtu.be'  in self.ui.plainTextEdit.toPlainText() or 'youtube.com' in self.ui.plainTextEdit.toPlainText():
            self.download_yt_vid()
        else:
            self.main.input_file = self.ui.plainTextEdit.toPlainText()
            self.main.localFile = True
            self.main.videoName = 'output.mp4'
            window.close()
            
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
    def download_yt_vid(self):
        try:
                print(self.ui.plainTextEdit.toPlainText())
                result = subprocess.run([f'{thisdir}/bin/yt-dlp_linux', '-F', self.ui.plainTextEdit.toPlainText()], capture_output=True, text=True)
                self.ui.qualityCombo.clear()
                if result.returncode == 0:
                    stdout_lines = result.stdout.splitlines()
                    resolutions_list = []
                    self.dict_res_id_fps = {}
                    fps_list=[]
                    i=0
                    for line in reversed(stdout_lines):
                       
                        if 'mp4' in line:
                            
                            resolution = re.findall(r'[\d]*x[\d]*',line)
                            if len(resolution) > 0:
                                if resolution[0] not in resolutions_list:
                                    res=resolution[0]
                                    resolutions_list.append(res)
                                    id=line[:3]
                                    fps=(line[22:24])
                                    self.dict_res_id_fps[res] = [id,fps]
                                    self.ui.qualityCombo.addItem(resolution[0])
                    self.duration = self.get_youtube_video_duration(self.ui.plainTextEdit.toPlainText())
                    self.ui.error_label.clear()
                    self.main.input_file = f'{thisdir}/{self.get_youtube_video_name(self.ui.plainTextEdit.toPlainText())}.mp4'
                    self.main.videoName = f'{self.get_youtube_video_name(self.ui.plainTextEdit.toPlainText())}.mp4'
                    self.ui.qualityLabel.show()
                    self.ui.qualityCombo.show()
                    self.ui.next.clicked.disconnect(self.next)
                    self.ui.next.clicked.connect(self.gen_youtubedlp_command)
                else:
                    if result.stderr == f"""ERROR: [generic] None: '{self.ui.plainTextEdit.toPlainText()}' is not a valid URL. Set --default-search "ytsearch" (or run  yt-dlp "ytsearch:{self.ui.plainTextEdit.toPlainText()}" ) to search YouTube\n""": 
                        self.ui.error_label.setText("Invalid URL")
                    else:
                        self.ui.error_label.setText(result.stderr)
                    self.ui.qualityCombo.hide()
                    self.ui.qualityLabel.hide()
        except Exception as e:
                print(e)
                if result.stderr == f"""ERROR: [generic] None: '{self.ui.plainTextEdit.toPlainText()}' is not a valid URL. Set --default-search "ytsearch" (or run  yt-dlp "ytsearch:{self.ui.plainTextEdit.toPlainText()}" ) to search YouTube\n""": 
                        self.ui.error_label.setText("Invalid URL")
                else:
                        self.ui.error_label.setText(result.stderr)
                self.ui.qualityCombo.hide()
                self.ui.qualityLabel.hide()
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
    def gen_youtubedlp_command(self):
        global return_command
        
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