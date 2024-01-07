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


def initializeInterpolation(self,AI):#1st stage in preparing render, starts all worker threads
    try:
        settings = Settings()
        os.system(f'rm -rf "{settings.RenderDir}/{self.videoName}_temp/"')
        #self.ui.QueueButton.show()
        
        

        self.AI = AI
        
        self.setDisableEnable(True)
        
        if settings.DiscordRPC == 'Enabled':
            try:
                start_discordRPC(self,'Interpolating')
            except Exception as e:
                print('No Discord Installation')
                print(e)
        #set UI
        
        self.ui.ETAPreview.setText('ETA:')
        self.ui.processedPreview.setText('Files Processed:')
                
        self.rifeThread = QThread()
            # Step 3: Create a worker object
        model = self.ui.Rife_Model.currentText()
        #check if ensemble
        if self.ui.EnsembleCheckBox.isChecked() == True and 'rife' in AI:
            if os.path.exists(f'{settings.ModelDir}/rife/{model}-ensemble'):
                model+='-ensemble'
            else:
                ensembleModelDoesntExist(self)
                
        self.rifeWorker = workers.interpolation(self,model)        
        
        self.ui.logsPreview.clear()
        

        # Step 4: Move worker to the thread
        self.rifeWorker.moveToThread(self.rifeThread)
        # Step 5: Connect signals and slots
        self.rifeThread.started.connect(self.rifeWorker.finishRenderSetup)
        self.rifeWorker.finished.connect(self.rifeThread.quit)
        self.rifeWorker.finished.connect(self.rifeWorker.deleteLater)
        self.rifeThread.finished.connect(self.rifeThread.deleteLater)
        self.rifeWorker.log.connect(self.addLinetoLogs)
        self.rifeWorker.removelog.connect(self.removeLastLineInLogs)
        # Step 6: Start the thread
        self.rifeThread.start()
        
        self.runPB()
    except Exception as e:
        traceback_info = traceback.format_exc()
        log(f'ERROR: {e} {traceback_info}')
        self.showDialogBox(e)
def start_interpolation(self,AI): #command directly connected to the rife start button
    try:           
        if self.input_file != '':
            self.render='rife'
            has_enough_space,predicted_space,total_space = checks.check_if_enough_space(self.input_file,self.render,self.times)
            if self.input_file.count("'") > 0 or '"' in self.input_file:
                quotes(self)
                return 
            if has_enough_space:
                initializeInterpolation(self,AI)
            elif not_enough_storage(self,predicted_space,total_space):
                initializeInterpolation(self,AI)
            else:
                pass
                    
        else:
                no_input_file(self)
    
    except Exception as e:
        traceback_info = traceback.format_exc()
        log(f'ERROR: {e} {traceback_info}')
        self.showDialogBox(e)
            


        
        
        