
import subprocess
from PyQt5.QtWidgets import  QMainWindow
import src.getLinkVideo.get_vid_from_link as getLinkedVideo
from src.workers import *
from cv2 import VideoCapture, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT
from src.return_data import *
import requests
import src.thisdir
thisdir = src.thisdir.thisdir()
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
        cap = VideoCapture(video_link)
        if not cap.isOpened():
            self.ui.error_label.setText("Failed to open the video.")
        fps = cap.get(CAP_PROP_FPS)
        cap.release()
        return fps

    
    def next(self):
        self.ui.error_label.clear()
        self.ui.next.hide()
        if 'youtu.be'  in self.ui.plainTextEdit.text() or 'youtube.com' in self.ui.plainTextEdit.text():
            #gets data from youtube using youtubedlp
            self.run_ytdl_thread()
            self.main.youtubeFile = True
        else:
            #get data from link here
            
            self.main.youtubeFile = False
            self.input_file = self.ui.plainTextEdit.text()
            
            self.main.videoName = self.input_file.split('/')[-1]
            
            try:
                response = requests.head(self.input_file)
                response.raise_for_status()
                content_type = response.headers.get('Content-Type')
                if content_type and content_type.startswith('video/'):
                    pass
                else:
                    self.ui.error_label.setText(f"Invalid video format. Content-Type: {content_type}")
                    self.ui.next.show()
                    return False
            except requests.exceptions.RequestException as e:
                try:
                    if response.status_code != 403: # ignores 403 errors cause they mostly work even when the exception is raised :\
                        self.ui.error_label.setText(f'{e}')
                        self.ui.next.show()
                        return False
                except:
                    not_valid_link(self.main)
                    
            self.main.fps=VideoName.return_video_framerate(self.input_file)
            self.main.localFile=False
            self.main.showChangeInFPS(self.main.fps)
            self.main.fc = VideoName.return_video_frame_count(self.input_file)
            self.ytVidRes = self.ui.qualityCombo.currentText()
            
            self.main.download_youtube_video_command = self.input_file
            self.main.input_file = f'{thisdir}/{self.main.videoName}'
            window.close()
        
    def run_ytdl_thread(self):
        self.thread = QThread()
        self.worker = downloadVideo(self,self.ui.plainTextEdit.text())
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
        if result == f"""ERROR: [generic] None: '{self.ui.plainTextEdit.text()}' is not a valid URL. Set --default-search "ytsearch" (or run  yt-dlp "ytsearch:{self.ui.plainTextEdit.text()}" ) to search YouTube\n""": 
            self.ui.error_label.setText("Invalid URL")
        else:
            self.ui.error_label.setText(result)
        self.ui.qualityCombo.hide()
        self.ui.qualityLabel.hide()
    def get_youtube_video_duration(self,url):
        try:
            result = subprocess.run([f'{thisdir}/bin/yt-dlp_linux', self.ui.plainTextEdit.text(), '--get-duration'], capture_output=True, text=True)
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
                video_name = result.stdout.strip().replace("/",'')
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
        print(self.dict_res_id_fps)
    def gen_youtubedlp_command(self):
        self.input_file = self.input_file.replace("'",'') # i need to find where this is called to rename the input file
        self.input_file = self.input_file.replace('"','')
        self.main.download_youtube_video_command = (f'{thisdir}/bin/yt-dlp_linux -f {self.dict_res_id_fps[self.ui.qualityCombo.currentText()][0]} "{self.ui.plainTextEdit.text()}" -o "{self.input_file}" && {thisdir}/bin/yt-dlp_linux -f 140 "{self.ui.plainTextEdit.text()}"  -o {thisdir}/audio.m4a')
        self.main.fps=float(self.dict_res_id_fps[self.ui.qualityCombo.currentText()][1])
        self.main.localFile=False
        
        self.main.fc = int(self.main.fps*self.duration)
        self.main.ytVidRes = self.ui.qualityCombo.currentText()
        
        self.main.input_file = self.input_file
        self.main.showChangeInFPS(self.main.fps)
        self.main.addLinetoLogs(f"Input file: {self.input_file}")
        window.close()
def get_linked_video(self):
    global window
    window = GetLinkedWindow(self)
    window.show()
    

