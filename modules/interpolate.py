#This script creates a class that takes in params like "RealESRGAN or Rife", the model for the program,  the times of upscaling, and the path of the video, and the output path
# hz
import src.return_data as return_data
import os
from src.settings import *
import src.runAI.transition_detection
from src.return_data import *
from src.messages import *
from src.discord_rpc import *
from modules.commands import *
import src.thisdir
import src.workers as workers
from PyQt5.QtCore import  QThread
from src.log import log
import traceback
thisdir = src.thisdir.thisdir()
homedir = os.path.expanduser(r"~")


def run_start(self,AI):
    os.system(f'rm -rf "{self.render_folder}/{self.videoName}_temp/"')
    self.ui.QueueButton.show()
    
    

    self.AI = AI
    settings = Settings()
    self.setDisableEnable(True)
    
    if settings.DiscordRPC == 'Enabled':
        start_discordRPC(self,'Interpolating')
    #set UI
    
    self.ui.ETAPreview.setText('ETA:')
    self.ui.processedPreview.setText('Files Processed:')
            
    self.rifeThread = QThread()
        # Step 3: Create a worker object
       
    self.rifeWorker = workers.interpolation(self,self.ui.Rife_Model.currentText())        
    
    self.ui.logsPreview.clear()
    

    # Step 4: Move worker to the thread
    self.rifeWorker.moveToThread(self.rifeThread)
    # Step 5: Connect signals and slots
    self.rifeThread.started.connect(self.rifeWorker.start_Render)
    self.rifeWorker.finished.connect(self.rifeThread.quit)
    self.rifeWorker.finished.connect(self.rifeWorker.deleteLater)
    self.rifeThread.finished.connect(self.rifeThread.deleteLater)
    self.rifeWorker.log.connect(self.addLinetoLogs)
    self.rifeWorker.removelog.connect(self.removeLastLineInLogs)
    # Step 6: Start the thread
    self.rifeWorker.finished.connect(self.endRife)
    self.rifeThread.start()
    
    self.runPB()

def start_interpolation(self,AI): 
    try:           
        if self.input_file != '':
            self.render='rife'
            has_enough_space,predicted_space,total_space = checks.check_if_enough_space(self.input_file,self.render,self.times)
            
            if has_enough_space:
                run_start(self,AI)
            elif not_enough_storage(self,predicted_space,total_space):
                run_start(self,AI)
            else:
                pass
                    
        else:
                no_input_file(self)
    
    except Exception as e:
        traceback_info = traceback.format_exc()
        log(f'{e} {traceback_info}')
        self.showDialogBox(e)
            


        
        
        