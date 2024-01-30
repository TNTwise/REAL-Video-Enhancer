import src.programData.return_data as return_data
import os
from src.programData.settings import *
import src.runAI.transition_detection
from src.programData.return_data import *
from src.misc.messages import *
from src.runAI.discord_rpc import *
import os
from modules.commands import *
from cv2 import VideoCapture, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT
import modules.upscale as upscale
from src.misc.log import *
#this file changes the GUI aspects of the AI
thisdir = src.programData.thisdir.thisdir()
homedir = os.path.expanduser(r"~")
import src.programData.checks as checks
from PyQt5.QtCore import QThread

import src.runAI.workers as workers

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
            has_enough_output_space,predicted_output_space,total_output_space = checks.check_if_enough_space_output_disk(self.input_file,self.render,self.times)
            if not has_enough_output_space:
                if not_enough_output_storage(self,predicted_output_space,total_output_space):
                    initializeInterpolation(self,AI)
                else:
                    pass
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
            


        
        
        