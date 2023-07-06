import src.settings 
import os
import src.return_data
import subprocess
from src.return_data import *
class TransitionDetection:
    def __init__(self,input_file):
        self.settings = src.settings.Settings()
        self.render_directory = self.settings.RenderDir
        self.input_file = input_file
        self.videoName = src.return_data.VideoName.return_video_name(f'{input_file}')
        self.thisdir=os.getcwd()
        self.fps = src.return_data.VideoName.return_video_framerate(f'{input_file}')
        self.full_render_dir = f'{self.render_directory}/{self.videoName}_temp'
        
        src.return_data.ManageFiles.create_folder(f'{self.full_render_dir}')
        src.return_data.ManageFiles.create_folder(f'{self.full_render_dir}/transitions')
            # Change scene\,0.6 to edit how much scene detections it does, do this for both ffmpeg commands
    def find_timestamps(self):
        if self.settings.SceneChangeDetection != 'Off':
            # This will get the timestamps of the scene changes, and for every scene change timestamp, i can times it by the fps count to get its current frame, and after interpolation, double it and replace it and it -1 frame with the transition frame stored in the transitions folder
            
           
            os.system(f'ffmpeg -i "{self.input_file}" -filter_complex "select=\'gt(scene\,{self.settings.SceneChangeDetection})\',metadata=print" -vsync vfr -q:v 2 "{self.full_render_dir}/transitions/%07d.png"')
            ffmpeg_cmd = f'ffmpeg -i "{self.input_file}" -filter_complex "select=\'gt(scene\,{self.settings.SceneChangeDetection})\',metadata=print" -vsync vfr -q:v 2 "{self.full_render_dir}/transitions/%07d.png"' 
            
            output = subprocess.check_output(ffmpeg_cmd, shell=True, stderr=subprocess.STDOUT)
            
            # Decode the output as UTF-8 and split it into lines
            output_lines = output.decode("utf-8").split("\n")
                    # Create a list to store the timestamps
            timestamps = []

            # Iterate over the output lines and extract the timestamps
            for line in output_lines:
                        if "pts_time" in line:
                            timestamp = str(line.split("_")[3])
                            timestamp = str(timestamp.split(':')[1])
                            timestamps.append(timestamp)
                    
            self.timestamps = timestamps
        
        

    def get_frame_num(self,times,frames_subtracted=0):
        self.times=times
        if self.settings.SceneChangeDetection != 'Off':
            frame_list =[]
            for i in self.timestamps:
                frame = float(i) * float(self.fps)
                
                frame = round(frame)
                frame = int(frame)
                
                #subtract from frame for anime method too

                
                frame = frame - frames_subtracted
                
                frame_list.append(frame)
            self.frame_list = frame_list
            
            # This code is shit, i will have to fix later, i have no idea why it works
            filenames = os.listdir(f'{self.full_render_dir}/transitions/')
            sorted_filenames = sorted(filenames)
            file_num_list = []
            list1 = []
            list2 = []
            self.prevFrameList = []
            self.fileToCopyDict = []
            self.fileToCopyDict1 = []
            self.fileToCopyDict2 = []
            self.fileToCopyDict3 = []
            self.fileToCopyDict4 = []
            for i in self.frame_list:
                        
                        i = int(i) * 2
                        i = int(i) - 1
                        i = str(i)
                        i = i.zfill(8)
                        list2.append(i)
            for j in self.frame_list:
                        
                        j = int(j) * times
                        prev_file = j - 1
                        file_to_copy_to = str(prev_file - 1).zfill(8)
                        file_to_copy_to1 = str(prev_file - 2).zfill(8)
                        file_to_copy_to2 = str(prev_file - 3).zfill(8)
                        file_to_copy_to3 = str(prev_file - 4).zfill(8)
                        file_to_copy_to4 = str(prev_file - 5).zfill(8)
                        #I am literally braindead lol, cant think of anyhting better for this stupid system
                        j = str(j)
                        prev_file = str(prev_file)
                        j = j.zfill(8)
                        prev_file = prev_file.zfill(8)
                        list1.append(j)
                        self.fileToCopyDict.append(file_to_copy_to)
                        self.fileToCopyDict1.append(file_to_copy_to1)
                        self.fileToCopyDict2.append(file_to_copy_to2)
                        self.fileToCopyDict3.append(file_to_copy_to3)
                        self.fileToCopyDict4.append(file_to_copy_to4)
                        self.prevFrameList.append(prev_file)
                        self.list1 = list1
            
            
            
            p = 0
            o = 1
            os.chdir(f'{self.full_render_dir}/transitions/')
            for image in list1:
                
                
               
                os.system(f'mv "{self.full_render_dir}/transitions/{str(str(o).zfill(7))}.png" "{self.full_render_dir}/transitions/{list1[p]}{self.settings.Image_Type}"')
                # Commenting this out due to it overlaping frames os.system(f'cp "{self.render_directory}/{filename}/transitions/{list1[p]}{Image_Type}" "{self.render_directory}/{filename}/transitions/{list2[p]}{Image_Type}"')
                if times == 4 or times == 8:
                    os.system(f'cp "{self.full_render_dir}/transitions/{list1[p]}.png" "{self.full_render_dir}/transitions/{self.prevFrameList[p]}{self.settings.Image_Type}"')
                    os.system(f'cp "{self.full_render_dir}/transitions/{list1[p]}.png" "{self.full_render_dir}/transitions/{self.fileToCopyDict[p]}{self.settings.Image_Type}"')
                if times == 8:
                    os.system(f'cp "{self.full_render_dir}/transitions/{list1[p]}.png" "{self.full_render_dir}/transitions/{self.fileToCopyDict1[p]}{self.settings.Image_Type}"')
                    os.system(f'cp "{self.full_render_dir}/transitions/{list1[p]}.png" "{self.full_render_dir}/transitions/{self.fileToCopyDict2[p]}{self.settings.Image_Type}"')
                    os.system(f'cp "{self.full_render_dir}/transitions/{list1[p]}.png" "{self.full_render_dir}/transitions/{self.fileToCopyDict3[p]}{self.settings.Image_Type}"')
                    os.system(f'cp "{self.full_render_dir}/transitions/{list1[p]}.png" "{self.full_render_dir}/transitions/{self.fileToCopyDict4[p]}{self.settings.Image_Type}"')
                    #This is so dumb lmao, ik there is a better way but i am lazy lol
                
                p+=1
                o+=1
                # IK this is dumb. but i cant think of anything else rn
            '''if times == 4:
                    for file,copyto in self.fileToCopyDict.items():
                        os.system(f'cp "{self.full_render_dir}/input_frames/{file}.png" "{self.full_render_dir}/transitions/{copyto}.png"')'''
                  
                  
            os.chdir(f'{self.thisdir}/rife-vulkan-models')
    def merge_frames(self):
        p = 0
        o = 1
        
        os.chdir(f'{self.full_render_dir}/transitions/')
        for i in os.listdir():
            
                os.system(f'cp {i} "{self.full_render_dir}/output_frames/"')
        
        for image in self.frame_list:
            os.system(f'mv "{self.full_render_dir}/transitions/{self.list1[p]}.png" "{self.full_render_dir}/transitions/{str(str(o).zfill(7))}.png" ')
            p+=1
            o+=1
        os.chdir(f'{self.thisdir}/')