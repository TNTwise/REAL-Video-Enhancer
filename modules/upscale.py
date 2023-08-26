
import src.return_data as return_data
import os
from src.settings import *
import src.runAI.transition_detection
from src.return_data import *
from src.messages import *
from src.discord_rpc import *
import os
from modules.commands import *
from cv2 import VideoCapture, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT

import src.thisdir
thisdir = src.thisdir.thisdir()
homedir = os.path.expanduser(r"~")
from PyQt5.QtCore import  QThread
import src.workers as workers


    
def start_upscale(self,AI):
    try:
        if self.input_file != '':
            os.system(f'rm -rf "{self.render_folder}/{self.videoName}_temp/"')
            self.ui.QueueButton.show()
            self.render='esrgan'
            self.AI = AI
            settings = Settings()
            self.setDisableEnable(True)
            
            if settings.DiscordRPC == 'Enabled':
                start_discordRPC(self,'Upscaling')
                
            realESRGAN_Model = self.ui.Rife_Model.currentText()
            realESRGAN_Times = self.ui.Rife_Times.currentText()
            if AI == 'realesrgan-ncnn-vulkan':
                if realESRGAN_Model == 'Default':
                    self.realESRGAN_Model = '-n realesrgan-x4plus -s 4'
                if realESRGAN_Model == 'Animation':
                    self.realESRGAN_Model = f'-n realesr-animevideov3 -s {realESRGAN_Times}'
            self.ui.logsPreview.append(f'[Extracting Frames]')
            self.ui.ETAPreview.setText('ETA:')
            self.ui.processedPreview.setText('Files Processed:')
                    
            self.upscaleThread = QThread()
                # Step 3: Create a worker object
            
            self.upscaleWorker = workers.upscale(self)        
            

            

            # Step 4: Move worker to the thread
            self.upscaleWorker.moveToThread(self.upscaleThread)
            # Step 5: Connect signals and slots
            self.upscaleThread.started.connect(self.upscaleWorker.start_Render)
            self.upscaleWorker.finished.connect(self.upscaleThread.quit)
            self.upscaleWorker.finished.connect(self.upscaleWorker.deleteLater)
            self.upscaleThread.finished.connect(self.upscaleThread.deleteLater)
            self.upscaleWorker.log.connect(self.addLinetoLogs)
            self.upscaleWorker.removelog.connect(self.removeLastLineInLogs)
            # Step 6: Start the thread
            
            self.upscaleThread.start()
            self.runPB()
    except Exception as e:
        self.showDialogBox(e)