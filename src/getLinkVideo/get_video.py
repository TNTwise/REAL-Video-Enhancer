
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
        self.ui.qualityCombo.currentIndexChanged.connect(self.setRes)
        
        self.show()
    def setRes(self):
         
         self.res = self.ui.qualityCombo.currentText()
         print(self.res)
         self.ui.qualityCombo.setCurrentText(self.res)
    def next(self):
        
        if 'youtu.be'  in self.ui.plainTextEdit.toPlainText() or 'youtube.com' in self.ui.plainTextEdit.toPlainText():
            self.download_yt_vid()
        else:
            
            return 0

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
                    
                    
                    self.ui.error_label.clear()
                    
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

    def gen_youtubedlp_command(self):
        print(self.dict_res_id_fps[self.ui.qualityCombo.currentText()][0])
        global return_command
        self.main.extract_frames_from_youtube_video_command = (f'{thisdir}/bin/yt-dlp_linux -f {self.dict_res_id_fps[self.ui.qualityCombo.currentText()][0]} "{self.ui.plainTextEdit.toPlainText()}"  -o - | ffmpeg -i - -s {self.ui.qualityCombo.currentText()} out.mp4 && {thisdir}/bin/yt-dlp_linux -f 140 "{self.ui.plainTextEdit.toPlainText()}"  -o - | ffmpeg -i -  output.m4a')
        self.main.fps=int(self.dict_res_id_fps[self.ui.qualityCombo.currentText()][1])
        self.main.localFile=False
        self.main.showChangeInFPS(False)
        window.close()
def get_linked_video(self):
    global window
    window = GetLinkedWindow(self)
    window.show()
    

#./yt-dlp_linux -f best "https://www.youtube.com/watch?v=k9hv2l3NOZU"  -o - | ffmpeg -i - frames/%08d.jpg && ./yt-dlp_linux -f 140 "https://www.youtube.com/watch?v=k9hv2l3NOZU"  -o - | ffmpeg -i -  output.m4a