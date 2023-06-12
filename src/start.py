#This script creates a class that takes in params like "RealESRGAN or Rife", the model for the program,  the times of upscaling, and the path of the video, and the output path
# hz
import src.return_data as return_data
import os
from src.settings import *
import glob
from threading import Thread
import src.transition_detection
from src.return_data import *
from src.messages import *
thisdir= os.getcwd()
homedir = os.path.expanduser(r"~")
settings = Settings()
def start(renderdir,videoName,videopath):
        global fps
        fps = return_data.Fps.return_video_fps(fr'{videopath}')
        
        
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
        return output_video_file

def startRife(self): #should prob make this different, too similar to start_rife but i will  think of something later prob
        
    # Calculate the aspect ratio
                
        
        if self.input_file != '':
            # Calculate the aspect ratio
            videoName = VideoName.return_video_name(fr'{self.input_file}')
            self.videoName = videoName
            video = cv2.VideoCapture(self.input_file)
            self.videowidth = video.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.videoheight = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.aspectratio = self.videowidth / self.videoheight
            self.setDisableEnable(True)
            os.system(f'rm -rf "{self.render_folder}/{self.videoName}_temp/"')
            self.transitionDetection = src.transition_detection.TransitionDetection(self.input_file)
            self.times = int(self.ui.Rife_Times.currentText()[0])
            self.ui.logsPreview.append(f'Extracting Frames')
            
            if int(self.ui.Rife_Times.currentText()[0]) == 2:
                self.rifeThread = Thread(target=lambda: start_rife(self,(self.ui.Rife_Model.currentText().lower()),2,self.input_file,self.output_folder,1))
            if int(self.ui.Rife_Times.currentText()[0]) == 4:
                self.rifeThread = Thread(target=lambda: start_rife(self,(self.ui.Rife_Model.currentText().lower()),4,self.input_file,self.output_folder,2))
            if int(self.ui.Rife_Times.currentText()[0]) == 8:
                self.rifeThread = Thread(target=lambda: start_rife(self,(self.ui.Rife_Model.currentText().lower()),8,self.input_file,self.output_folder,3))
            self.rifeThread.start()
            self.runPB(self.videoName,self.times)
        else:
            no_input_file(self)

    
def start_rife(self,model,times,videopath,outputpath,end_iteration):
        
        
        self.fps = VideoName.return_video_framerate(f'{self.input_file}')
        self.ui.ETAPreview.setText('ETA:')
        self.ui.processedPreview.setText('Files Processed:')
        
        # Have to put this before otherwise it will error out ???? idk im not good at using qt.....
                
                
        #self.runLogs(videoName,times)
        self.transitionDetection.find_timestamps()
        self.transitionDetection.get_frame_num()
        start(self.render_folder,self.videoName,videopath)
        
        
        
        
        
                #change progressbar value
    
        
        
            
        Thread(target=self.calculateETA).start()
        input_frames = len(os.listdir(f'{self.render_folder}/{self.videoName}_temp/input_frames/'))
        os.system(f'"{thisdir}/rife-vulkan-models/rife-ncnn-vulkan" -n {input_frames*times}  -m  {model} -i "{self.render_folder}/{self.videoName}_temp/input_frames/" -o "{self.render_folder}/{self.videoName}_temp/output_frames/" -j 10:10:10 ')
        
        if os.path.exists(f'{self.render_folder}/{self.videoName}_temp/output_frames/') == False or os.path.isfile(f'{self.render_folder}/{self.videoName}_temp/audio.m4a') == False:
            show_on_no_output_files(self)
        else:
            self.transitionDetection.merge_frames()
            from time import sleep
            sleep(1)
            self.output_file = end(self.render_folder,self.videoName,videopath,times,outputpath, self.videoQuality,self.encoder)
            

def endRife(self):
        
        self.addLinetoLogs(f'Finished! Output video: {self.output_file}\n')
        self.setDisableEnable(False)
        self.ui.RifePB.setValue(self.ui.RifePB.maximum())
        self.ui.ETAPreview.setText('ETA: 00:00:00')
        self.ui.imagePreview.clear()
        self.ui.processedPreview.setText(f'Files Processed: {self.fileCount} / {self.fileCount}')
        self.ui.imageSpacerFrame.show()
