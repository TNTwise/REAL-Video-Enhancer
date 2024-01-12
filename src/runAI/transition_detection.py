import os
import src.return_data
import subprocess
from src.return_data import *
from src.settings import *
def generate_opposite_pair(number, start, end):
    if number < start or number > end:
        return None  # Number is outside the specified range

    opposite = end - (number - start)
    return opposite
import math

class TransitionDetection:
    def __init__(self,originalSelf):
        self.settings = Settings()
        self.render_directory = self.settings.RenderDir
        self.input_file = originalSelf.input_file
        self.videoName = originalSelf.videoName
        import src.thisdir
        #self.main = originalSelf
        self.thisdir = src.thisdir.thisdir()
        self.fps = originalSelf.fps
        self.full_render_dir = f'{self.render_directory}/{self.videoName}_temp'
        self.main = originalSelf
        src.return_data.ManageFiles.create_folder(f'{self.full_render_dir}')
        src.return_data.ManageFiles.create_folder(f'{self.full_render_dir}/transitions')
            # Change scene\,0.6 to edit how much scene detections it does, do this for both ffmpeg commands
    def find_timestamps(self):
        if self.settings.SceneChangeDetection != 'Off':
            # This will get the timestamps of the scene changes, and for every scene change timestamp, i can times it by the fps count to get its current frame, and after interpolation, double it and replace it and it -1 frame with the transition frame stored in the transitions folder
            try:
                os.mkdir(f"{self.full_render_dir}/transitions/")
            except:
                 pass
            #self.main.addLinetoLogs('Detecting Transitions')
            
            if self.settings.Image_Type != '.webp':
                ffmpeg_cmd = f'"{thisdir}/bin/ffmpeg" -i "{self.input_file}" -filter_complex "select=\'gt(scene\,{self.settings.SceneChangeDetection})\',metadata=print" -vsync vfr -q:v 1 "{self.full_render_dir}/transitions/%07d{self.settings.Image_Type}"' 
            else:
                ffmpeg_cmd = f'"{thisdir}/bin/ffmpeg" -i "{self.input_file}" -filter_complex "select=\'gt(scene\,{self.settings.SceneChangeDetection})\',metadata=print" -vsync vfr -q:v 100 "{self.full_render_dir}/transitions/%07d.png"' 
            output = subprocess.check_output(ffmpeg_cmd, shell=True, stderr=subprocess.STDOUT)
            #self.main.addLinetoLogs(f'Transitions detected: {len(os.listdir(f"{self.full_render_dir}/transitions/"))}')
            # Decode the output as UTF-8 and split it into lines
            output_lines = output.decode("utf-8").split("\n")
                    # Create a list to store the timestamps
            timestamps = []

            # Iterate over the output lines and extract the timestamps
            for line in output_lines:
                        if "pts_time" in line:
                            timestamp = str(line.split("_")[3])
                            timestamp = str(timestamp.split(':')[1])
                            timestamps.append(math.ceil(round(float(timestamp)*float(self.fps))*self.main.times))
                    
            self.timestamps = timestamps
        

    def get_frame_num(self,times,frames_subtracted=0):
        self.times=times
        settings = Settings()
        try:
            
            if self.settings.SceneChangeDetection != 'Off':
                transitions = os.listdir(f'{self.full_render_dir}/transitions/')
                if not os.path.exists(f'{self.full_render_dir}/transitions/temp/'): os.mkdir(f'{self.full_render_dir}/transitions/temp/')
                for iteration,i in enumerate(transitions):
                    if settings.Image_Type != '.webp':
                                os.system(f'mv "{self.full_render_dir}/transitions/{str(str(iteration+1).zfill(7))}{settings.Image_Type}" "{self.full_render_dir}/transitions/temp/{self.timestamps[iteration]}{settings.Image_Type}"')
                    else:
                                os.system(f'mv "{self.full_render_dir}/transitions/{str(str(iteration+1).zfill(7))}.png" "{self.full_render_dir}/transitions/temp/{self.timestamps[iteration]}{settings.Image_Type}"')
                for i in self.timestamps:
                        for j in range(math.ceil(times)):
                                os.system(f'cp "{self.full_render_dir}/transitions/temp/{i}{settings.Image_Type}" "{self.full_render_dir}/transitions/{str(int(i)-j).zfill(8)}{settings.Image_Type}"' )
                os.system(f'rm -rf "{self.full_render_dir}/transitions/temp/"')
        except Exception as e:
            tb = traceback.format_exc()
            log(e,tb)
            print(e,tb)
            
            
                        
    def merge_frames(self,iteration=None):
        if iteration == None:  
            #os.system(f'cp "{self.full_render_dir}/transitions/"* "{self.full_render_dir}/output_frames/0/"')
            
            os.chdir(f'{self.full_render_dir}/transitions/')
            for i in os.listdir():
                if os.path.isfile(f'{self.full_render_dir}/output_frames/0/{i}'):
                    os.system(f'cp {i} "{self.full_render_dir}/output_frames/0/"')
            
            
            os.chdir(f'{self.thisdir}/')
        else:
            
            for i in os.listdir(f"{self.full_render_dir}/transitions/"):
                if os.path.isfile(f'{self.full_render_dir}/output_frames/{i}.mp4'):
                    try:
                        os.removedirs(f"{self.full_render_dir}/transitions/{i}")# i think i can use this instead of rm
                    except:
                        os.system(f'rm -rf "{self.full_render_dir}/transitions/{i}"')
                        
            os.system(f'cp -r "{self.full_render_dir}/transitions/{iteration}"* "{self.full_render_dir}/output_frames/0/"')