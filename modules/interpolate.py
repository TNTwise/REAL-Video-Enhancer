#This script creates a class that takes in params like "RealESRGAN or Rife", the model for the program,  the times of upscaling, and the path of the video, and the output path
# hz
import src.return_data as return_data
import os
from src.settings import *
import glob
from threading import Thread
import src.runAI.transition_detection
from src.return_data import *
from src.messages import *
from src.discord_rpc import *
import glob
import os
from modules.commands import *
import src.thisdir
import src.workers as workers
from PyQt5.QtCore import QObject, QThread, pyqtSignal
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
    self.ui.logsPreview.append(f'[Extracting Frames]')
    self.ui.ETAPreview.setText('ETA:')
    self.ui.processedPreview.setText('Files Processed:')
            
    self.rifeThread = QThread()
        # Step 3: Create a worker object
       
    self.rifeWorker = workers.interpolation(self,self.ui.Rife_Model.currentText().lower())        
    

    

    # Step 4: Move worker to the thread
    self.rifeWorker.moveToThread(self.rifeThread)
    # Step 5: Connect signals and slots
    self.rifeThread.started.connect(self.rifeWorker.start_Render)
    self.rifeWorker.finished.connect(self.rifeThread.quit)
    self.rifeWorker.finished.connect(self.rifeWorker.deleteLater)
    self.rifeThread.finished.connect(self.rifeThread.deleteLater)
    self.rifeWorker.log.connect(self.addLinetoLogs)
    # Step 6: Start the thread
    
    self.rifeThread.start()
    
    self.runPB()

def start_interpolation(self,AI): #should prob make this different, too similar to start_rife but i will  think of something later prob
    
    # Calculate the aspect ratio
                
        
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
             
            

    
def start_Render(self,model,times,videopath,outputpath):
        
        
        
        self.ui.ETAPreview.setText('ETA:')
        self.ui.processedPreview.setText('Files Processed:')
        
        # Have to put this before otherwise it will error out ???? idk im not good at using qt.....
                
                
        #self.runLogs(videoName,times)
        start(self,self.render_folder,self.videoName,videopath,times)
        self.transitionDetection = src.runAI.transition_detection.TransitionDetection(self)
        self.transitionDetection.find_timestamps()
        self.transitionDetection.get_frame_num(times)
        self.endNum = 0 # This variable keeps track of the amound of zeros to fill in the output frames, this helps with pausing and resuming so rife wont overwrite the original frames.
        Render(self,model,times,videopath,outputpath)
        
        
        
        
        
    
        
def Render(self,model,times,videopath,outputpath):   
        self.paused = False
        settings=Settings()
        #Thread(target=self.calculateETA).start()
        input_frames = len(os.listdir(f'{self.render_folder}/{self.videoName}_temp/input_frames/'))
        if self.AI == 'rife-ncnn-vulkan':
            if model == 'rife-v4.6' or model == 'rife-v4':
                os.system(f'"{settings.ModelDir}/rife/rife-ncnn-vulkan" -n {input_frames*times}  -m  {model} -i "{self.render_folder}/{self.videoName}_temp/input_frames/" -o "{self.render_folder}/{self.videoName}_temp/output_frames/" {return_gpu_settings(self)} -f %08d{self.settings.Image_Type}')
            else:
                os.system(f'"{settings.ModelDir}/rife/rife-ncnn-vulkan"  -m  {model} -i "{self.render_folder}/{self.videoName}_temp/input_frames/" -o "{self.render_folder}/{self.videoName}_temp/output_frames/" {return_gpu_settings(self)} -f %08d{self.settings.Image_Type} ')
        if os.path.exists(f'{self.render_folder}/{self.videoName}_temp/output_frames/') == False:
             show_on_no_output_files(self)
        
        else:
            self.transitionDetection.merge_frames()
            
            self.output_file = end(self,self.render_folder,self.videoName,videopath,times,outputpath, self.videoQuality,self.encoder)
 