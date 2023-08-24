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


class TransitionDetection:
    def __init__(self,originalSelf):
        self.settings = Settings()
        self.render_directory = self.settings.RenderDir
        self.input_file = originalSelf.input_file
        self.videoName = originalSelf.videoName
        import src.thisdir
        self.thisdir = src.thisdir.thisdir()
        self.fps = originalSelf.fps
        self.full_render_dir = f'{self.render_directory}/{self.videoName}_temp'
        
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
            if self.settings.Image_Type != '.webp':
                ffmpeg_cmd = f'{thisdir}/bin/ffmpeg -i "{self.input_file}" -filter_complex "select=\'gt(scene\,{self.settings.SceneChangeDetection})\',metadata=print" -vsync vfr -q:v 1 "{self.full_render_dir}/transitions/%07d{self.settings.Image_Type}"' 
            else:
                 ffmpeg_cmd = f'{thisdir}/bin/ffmpeg -i "{self.input_file}" -filter_complex "select=\'gt(scene\,{self.settings.SceneChangeDetection})\',metadata=print" -vsync vfr -q:v 100 "{self.full_render_dir}/transitions/%07d.png"' 
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
            print(self.timestamps)
        

    def get_frame_num(self,times,frames_subtracted=0):
        self.times=times
        settings = Settings()
        if self.settings.SceneChangeDetection != 'Off':
            frame_list =[]
            for i in self.timestamps:
                print(self.fps)
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
            print(list1)
            for image in list1:
                
                
                if settings.Image_Type != '.webp':
                    os.system(f'mv "{self.full_render_dir}/transitions/{str(str(o).zfill(7))}{self.settings.Image_Type}" "{self.full_render_dir}/transitions/{list1[p]}{self.settings.Image_Type}"')
                else:
                    os.system(f'mv "{self.full_render_dir}/transitions/{str(str(o).zfill(7))}.png" "{self.full_render_dir}/transitions/{list1[p]}{self.settings.Image_Type}"')
                # Commenting this out due to it overlaping frames os.system(f'cp "{self.render_directory}/{filename}/transitions/{list1[p]}{Image_Type}" "{self.render_directory}/{filename}/transitions/{list2[p]}{Image_Type}"')
                if times == 4 or times == 8:
                    os.system(f'cp "{self.full_render_dir}/transitions/{list1[p]}{self.settings.Image_Type}" "{self.full_render_dir}/transitions/{self.prevFrameList[p]}{self.settings.Image_Type}"')
                    os.system(f'cp "{self.full_render_dir}/transitions/{list1[p]}{self.settings.Image_Type}" "{self.full_render_dir}/transitions/{self.fileToCopyDict[p]}{self.settings.Image_Type}"')
                if times == 8:
                    os.system(f'cp "{self.full_render_dir}/transitions/{list1[p]}{self.settings.Image_Type}" "{self.full_render_dir}/transitions/{self.fileToCopyDict1[p]}{self.settings.Image_Type}"')
                    os.system(f'cp "{self.full_render_dir}/transitions/{list1[p]}{self.settings.Image_Type}" "{self.full_render_dir}/transitions/{self.fileToCopyDict2[p]}{self.settings.Image_Type}"')
                    os.system(f'cp "{self.full_render_dir}/transitions/{list1[p]}{self.settings.Image_Type}" "{self.full_render_dir}/transitions/{self.fileToCopyDict3[p]}{self.settings.Image_Type}"')
                    os.system(f'cp "{self.full_render_dir}/transitions/{list1[p]}{self.settings.Image_Type}" "{self.full_render_dir}/transitions/{self.fileToCopyDict4[p]}{self.settings.Image_Type}"')
                    #This is so dumb lmao, ik there is a better way but i am lazy lol
                  
                       
                p+=1
                o+=1
                # IK this is dumb. but i cant think of anything else rn
            '''if times == 4:
                    for file,copyto in self.fileToCopyDict.items():
                        os.system(f'cp "{self.full_render_dir}/input_frames/{file}{self.settings.Image_Type}" "{self.full_render_dir}/transitions/{copyto}{self.settings.Image_Type}"')'''
                  
            if settings.RenderType == 'Optimized':# this will sort out the images into the correct directories
                    frame_count = VideoName.return_video_frame_count(self.input_file)
                    interpolation_sessions = ceildiv(int(frame_count*times),100)
                    
                    for i in range(interpolation_sessions):
                        os.mkdir(f'{self.settings.RenderDir}/{self.videoName}_temp/transitions/{i}')
                    for i in os.listdir(f'{self.full_render_dir}/transitions/'):
                        if settings.Image_Type in i:
                            frame_num = int(i.replace(settings.Image_Type,''))
                            file_to_move_to = int(ceildiv(frame_num,100))# frame increments in workers.py, too lazy to get data from there lol   
                            
                            os.system(f'mv "{self.full_render_dir}/transitions/{i}" "{self.full_render_dir}/transitions/{generate_opposite_pair(file_to_move_to,0,interpolation_sessions)}/"')    
                    files = os.listdir(f'{self.full_render_dir}/transitions/')
                    files.sort()
                        
    def merge_frames(self,frames_per_output_file=None):
        if frames_per_output_file == None:  
            p = 0
            o = 1
            
            os.chdir(f'{self.full_render_dir}/transitions/')
            for i in os.listdir():
                
                    os.system(f'cp {i} "{self.full_render_dir}/output_frames/"')
            
            for image in self.frame_list:
                os.system(f'mv "{self.full_render_dir}/transitions/{self.list1[p]}{self.settings.Image_Type}" "{self.full_render_dir}/transitions/{str(str(o).zfill(7))}{self.settings.Image_Type}" ')
                p+=1
                o+=1
            os.chdir(f'{self.thisdir}/')
        else:
            os.system(f'cp -r "{self.full_render_dir}/transitions/"* "{self.full_render_dir}/output_frames/"')