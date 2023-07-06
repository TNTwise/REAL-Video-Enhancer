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

thisdir= os.getcwd()
homedir = os.path.expanduser(r"~")



def start(renderdir,videoName,videopath,times):
        global fps
        fps = return_data.Fps.return_video_fps(fr'{videopath}')
        '''with open(f'{renderdir}/{videoName}_temp/data.txt', 'w') as f:
            f.write(f'{times}')'''
        return_data.ManageFiles.create_folder(f'{renderdir}/{videoName}_temp/')
        return_data.ManageFiles.create_folder(f'{renderdir}/{videoName}_temp/input_frames')
       
        
        
        os.system(f'ffmpeg -i "{videopath}" "{renderdir}/{videoName}_temp/input_frames/%08d.png" -y ') # Add image extraction setting here, also add ffmpeg command here as if its compiled or not
        os.system(f'ffmpeg -i "{videopath}" -vn -c:a aac -b:a 320k "{renderdir}/{videoName}_temp/audio.m4a" -y') # do same here i think maybe
        return_data.ManageFiles.create_folder(f'{renderdir}/{videoName}_temp/output_frames') # this is at end due to check in progressbar to start, bad implementation should fix later....

def end(renderdir,videoName,videopath,times,outputpath,videoQuality,encoder):
        
        
        
        if outputpath == '':
                outputpath = homedir
        if return_data.ManageFiles.isfile(f'{outputpath}/{videoName}_{int(fps*times)}fps.mp4') == True:
                i=1
                while return_data.ManageFiles.isfile(f'{outputpath}/{videoName}_{int(fps*times)}fps({i}).mp4') == True:
                        i+=1
                output_video_file = f'{outputpath}/{videoName}_{int(fps*times)}fps({i}).mp4' 

        else:
               output_video_file = f'{outputpath}/{videoName}_{int(fps*times)}fps.mp4' 
        
        os.system(f'ffmpeg -framerate {fps*times} -i "{renderdir}/{videoName}_temp/output_frames/%08d.png" -i "{renderdir}/{videoName}_temp/audio.m4a" -c:v libx{encoder} -crf {videoQuality} -c:a copy  -pix_fmt yuv420p "{output_video_file}" -y') #ye we gonna have to add settings up in this bish
        os.system(f'rm -rf "{renderdir}/{videoName}_temp/audio.m4a"')
        
        os.system(f'rm -rf "{renderdir}/{videoName}_temp/"')
        os.chdir(thisdir)
        return output_video_file

def startRife(self): #should prob make this different, too similar to start_rife but i will  think of something later prob
    
    # Calculate the aspect ratio
                
        
        if self.input_file != '':
            self.ui.QueueButton.show()
            self.render='rife'
            self.fps = VideoName.return_video_framerate(f'{self.input_file}')
            settings = Settings()
            # Calculate the aspect ratio
            videoName = VideoName.return_video_name(fr'{self.input_file}')
            self.videoName = videoName
            video = cv2.VideoCapture(self.input_file)
            self.videowidth = video.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.videoheight = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.aspectratio = self.videowidth / self.videoheight
            self.setDisableEnable(True)
            if settings.DiscordRPC == 'Enabled':
                start_discordRPC(self)
            os.system(f'rm -rf "{self.render_folder}/{self.videoName}_temp/"')
            self.transitionDetection = src.runAI.transition_detection.TransitionDetection(self.input_file)
            self.times = int(self.ui.Rife_Times.currentText()[0])
            self.ui.logsPreview.append(f'Extracting Frames')
            
            if int(self.ui.Rife_Times.currentText()[0]) == 2:
                self.rifeThread = Thread(target=lambda: start_rife(self,(self.ui.Rife_Model.currentText().lower()),2,self.input_file,self.output_folder,1))
            if int(self.ui.Rife_Times.currentText()[0]) == 4:
                self.rifeThread = Thread(target=lambda: start_rife(self,(self.ui.Rife_Model.currentText().lower()),4,self.input_file,self.output_folder,2))
            if int(self.ui.Rife_Times.currentText()[0]) == 8:
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
        start(self.render_folder,self.videoName,videopath,times)
        self.transitionDetection.find_timestamps()
        self.transitionDetection.get_frame_num(times)
        self.endNum = 0 # This variable keeps track of the amound of zeros to fill in the output frames, this helps with pausing and resuming so rife wont overwrite the original frames.
        Rife(self,model,times,videopath,outputpath,end_iteration)
        
        
        
        
        
                #change progressbar value
    
        
def Rife(self,model,times,videopath,outputpath,end_iteration):   
        self.paused = False
        #Thread(target=self.calculateETA).start()
        input_frames = len(os.listdir(f'{self.render_folder}/{self.videoName}_temp/input_frames/'))
        if model == 'rife-v4.6' or model == 'rife-v4':
            os.system(f'"{thisdir}/rife-vulkan-models/rife-ncnn-vulkan" -n {input_frames*times}  -m  {model} -i "{self.render_folder}/{self.videoName}_temp/input_frames/" -o "{self.render_folder}/{self.videoName}_temp/output_frames/" {return_gpu_settings(self)} -f %08d.png')
        else:
              os.system(f'"{thisdir}/rife-vulkan-models/rife-ncnn-vulkan"  -m  {model} -i "{self.render_folder}/{self.videoName}_temp/input_frames/" -o "{self.render_folder}/{self.videoName}_temp/output_frames/" {return_gpu_settings(self)} -f %08d.png ')
        if os.path.exists(f'{self.render_folder}/{self.videoName}_temp/output_frames/') == False or os.path.isfile(f'{self.render_folder}/{self.videoName}_temp/audio.m4a') == False:
            show_on_no_output_files(self)
        else:
            if self.paused == False:
                
                files=os.listdir(f'{self.render_folder}/{self.videoName}_temp/output_frames')
            
                files = sorted(files)
                iteration=1
                for i in files:
                    new_file = str(iteration).zfill(8)
                    os.rename(f'{self.render_folder}/{self.videoName}_temp/output_frames/{i}',f'{self.render_folder}/{self.videoName}_temp/output_frames/{new_file}.png')
                    iteration+=1 # fixes any files that were created from a pause/resume, and will fit them into a 8 digit file so ffmpeg can read them
                self.transitionDetection.merge_frames()
                
                self.output_file = end(self.render_folder,self.videoName,videopath,times,outputpath, self.videoQuality,self.encoder)
            else:
                pass


def renderRealsr(self):
    
    start(self.render_folder,self.videoName,self.input_file,1)
    os.chdir(f'{thisdir}/realesrgan-vulkan-models')
    realESRGAN(self)
def realESRGAN(self):
        self.endNum=0
        self.paused=False
        os.system(f'./realesrgan-ncnn-vulkan {self.realESRGAN_Model} -i "{self.render_folder}/{self.videoName}_temp/input_frames" -o "{self.render_folder}/{self.videoName}_temp/output_frames" {return_gpu_settings(self)} ')
        if os.path.exists(f'{self.render_folder}/{self.videoName}_temp/output_frames/') == False or os.path.isfile(f'{self.render_folder}/{self.videoName}_temp/audio.m4a') == False:
                show_on_no_output_files(self)
        else:
                if self.paused == False:
                    self.output_file = end(self.render_folder,self.videoName,self.input_file,1,self.output_folder, self.videoQuality,self.encoder)
                else:
                    pass
    
def startRealSR(self):
    if self.input_file != '':
        self.ui.QueueButton.show()
        
        self.render='esrgan'
        settings = Settings()
        self.ui.ETAPreview.setText('ETA:')
        self.ui.processedPreview.setText('Files Processed:')
        self.setDisableEnable(True)
        self.times = 1
        self.fps=VideoName.return_video_framerate(f'{self.input_file}')
        
        video = cv2.VideoCapture(self.input_file)
        self.videowidth = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.videoheight = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.aspectratio = self.videowidth / self.videoheight
        self.setDisableEnable(True)
        
        if settings.DiscordRPC == 'Enabled':
            start_discordRPC(self)
        os.system(f'rm -rf "{self.render_folder}/{self.videoName}_temp/"')
        
        os.system(f'rm -rf "{self.render_folder}/{self.videoName}_temp/"')
        realESRGAN_Model = self.ui.RealESRGAN_Model.currentText()
        realESRGAN_Times = self.ui.RealESRGAN_Times.currentText()
        if realESRGAN_Model == 'Default':
            self.realESRGAN_Model = '-n realesrgan-x4plus -s 4'
        if realESRGAN_Model == 'Animation':
            self.realESRGAN_Model = f'-n realesr-animevideov3 -s {realESRGAN_Times}'
        Thread(target=lambda: renderRealsr(self)).start()
        self.runPB()
    else:
         no_input_file(self)