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
thisdir= os.getcwd()
homedir = os.path.expanduser(r"~")




def startRife(self): #should prob make this different, too similar to start_rife but i will  think of something later prob
    
    # Calculate the aspect ratio
                
        
        if self.input_file != '':
            self.ui.QueueButton.show()
            self.render='rife'
            
            settings = Settings()
            self.setDisableEnable(True)
            
            if settings.DiscordRPC == 'Enabled':
                start_discordRPC(self,'Interpolating')
            self.transitionDetection = src.runAI.transition_detection.TransitionDetection(self.input_file)
            self.ui.logsPreview.append(f'Extracting Frames')
            
            if self.times == 2:
                self.rifeThread = Thread(target=lambda: start_rife(self,(self.ui.Rife_Model.currentText().lower()),2,self.input_file,self.output_folder,1))
            if self.times == 4:
                self.rifeThread = Thread(target=lambda: start_rife(self,(self.ui.Rife_Model.currentText().lower()),4,self.input_file,self.output_folder,2))
            if self.times == 8:
                self.rifeThread = Thread(target=lambda: start_rife(self,(self.ui.Rife_Model.currentText().lower()),8,self.input_file,self.output_folder,3))
            self.rifeThread.start()
            self.runPB()
        else:
            no_input_file(self)

    
def start_rife(self,model,times,videopath,outputpath,end_iteration):
        
        
        
        self.ui.ETAPreview.setText('ETA:')
        self.ui.processedPreview.setText('Files Processed:')
        
        # Have to put this before otherwise it will error out ???? idk im not good at using qt.....
                
                
        #self.runLogs(videoName,times)
        start(self,self.render_folder,self.videoName,videopath,times)
        self.transitionDetection.find_timestamps()
        self.transitionDetection.get_frame_num(times)
        self.endNum = 0 # This variable keeps track of the amound of zeros to fill in the output frames, this helps with pausing and resuming so rife wont overwrite the original frames.
        Rife(self,model,times,videopath,outputpath,end_iteration)
        
        
        
        
        
                #change progressbar value
    
        
def Rife(self,model,times,videopath,outputpath,end_iteration):   
        self.paused = False
        settings=Settings()
        #Thread(target=self.calculateETA).start()
        input_frames = len(os.listdir(f'{self.render_folder}/{self.videoName}_temp/input_frames/'))
        if model == 'rife-v4.6' or model == 'rife-v4':
            os.system(f'"{settings.ModelDir}/rife/rife-ncnn-vulkan" -n {input_frames*times}  -m  {model} -i "{self.render_folder}/{self.videoName}_temp/input_frames/" -o "{self.render_folder}/{self.videoName}_temp/output_frames/" {return_gpu_settings(self)} -f %08d{self.settings.Image_Type}')
        else:
              os.system(f'"{settings.ModelDir}/rife/rife-ncnn-vulkan"  -m  {model} -i "{self.render_folder}/{self.videoName}_temp/input_frames/" -o "{self.render_folder}/{self.videoName}_temp/output_frames/" {return_gpu_settings(self)} -f %08d{self.settings.Image_Type} ')
        if os.path.exists(f'{self.render_folder}/{self.videoName}_temp/output_frames/') == False or os.path.isfile(f'{self.render_folder}/{self.videoName}_temp/audio.m4a') == False:
            show_on_no_output_files(self)
        else:
            if self.paused == False:
                #Scraping this for now
                '''files=os.listdir(f'{self.render_folder}/{self.videoName}_temp/output_frames')
            
                files = sorted(files)
                iteration=1
                for i in files:
                    new_file = str(iteration).zfill(8)
                    os.rename(f'{self.render_folder}/{self.videoName}_temp/output_frames/{i}',f'{self.render_folder}/{self.videoName}_temp/output_frames/{new_file}{self.settings.Image_Type}')
                    iteration+=1 # fixes any files that were created from a pause/resume, and will fit them into a 8 digit file so ffmpeg can read them'''
                self.transitionDetection.merge_frames()
                
                self.output_file = end(self,self.render_folder,self.videoName,videopath,times,outputpath, self.videoQuality,self.encoder)
            else:
                pass


