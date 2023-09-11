
#This script creates a class that takes in params like "RealESRGAN or Rife", the model for the program,  the times of upscaling, and the path of the video, and the output path
# hz
import src.return_data as return_data
import os
from src.settings import *
from threading import Thread
import src.runAI.transition_detection
from src.return_data import *
from src.messages import *
from src.discord_rpc import *
import requests
import re
import src.thisdir
import src.checks as checks
thisdir = src.thisdir.thisdir()
from src.log import log
import traceback
def return_gpu_settings(self):
    if int(self.gpuMemory) < 1:
        gpu_usage = f'-j 1:1:1'
    else:
        num = int(int(self.gpuMemory))
        gpu_usage = f'-j {num}:{num}:{num}'
    return gpu_usage
def print_output(thread,self,extracting,pipe):
    total_frame_count = VideoName.return_video_frame_count(self.input_file)
    
    
    mode='Merged'
    if extracting == True:
           mode = 'Extracted'
           times=1
    if mode == 'Merged':
           times = self.times
    thread.log.emit(" ")
    while True:
        line = pipe.readline()
        if not line:
            break
        else:
                if  'frame' in line:
                        frame_num = re.findall(r'frame=[ ]*[\d]*',line)
                        if len(frame_num) != 0:
                                thread.removelog.emit(f'Frames {mode}:')
                                frame_num = frame_num[0]
                                
                                frame_num = frame_num.split('=')[1]
                                thread.log.emit(f"Frames {mode}: {frame_num} / {int(total_frame_count*times)}")
def run_subprocess_with_realtime_output(thread,self,command,extracting=False):
    self.ffmpeg  = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
        bufsize=1,  # Line-buffered output
        universal_newlines=True  # Ensure newline translation
    )

    stdout_thread = Thread(target=print_output, args=(thread,self,extracting,self.ffmpeg.stdout,))
    stderr_thread = Thread(target=print_output, args=(thread,self,extracting,self.ffmpeg.stderr,))

    stdout_thread.start()
    stderr_thread.start()

    # Wait for the process to finish
    self.ffmpeg.wait()

    # Wait for the output threads to finish printing
    stdout_thread.join()
    stderr_thread.join()

    return self.ffmpeg.returncode

def get_video_from_link(self,thread):
        if self.youtubeFile == True:
                thread.log.emit("[Downloading YouTube Video]")
                os.system(f'{self.download_youtube_video_command}')
                
        else:
                thread.log.emit("[Downloading Video]")
                response = requests.get(self.download_youtube_video_command, stream=True)
                
                # Check if the download was successful
                if response.status_code != 200:
                        raise Exception(f"Failed to download the file. Status code: {response.status_code}")
                
                with open(f'{thisdir}/{self.videoName}', 'wb') as file:
                        for chunk in response.iter_content(chunk_size=8192):
                                file.write(chunk)

def start(thread,self,renderdir,videoName,videopath,times):
        try:
                self.file_drop_widget.hide()
                # i need to clean this up lol
                os.system(f'rm -rf "{self.render_folder}/{self.videoName}_temp/"')
                #Gets the width and height
                global fps
                global height
                global width
                if self.localFile == False:
                        get_video_from_link(self,thread)

                self.fps = VideoName.return_video_framerate(f'{self.input_file}')
                settings = Settings()
                # Calculate the aspect ratio
                videoName = VideoName.return_video_name(fr'{self.input_file}')
                self.videoName = videoName
                video = cv2.VideoCapture(self.input_file)
                try:
                        self.videowidth = video.get(cv2.CAP_PROP_FRAME_WIDTH)
                        
                        self.videoheight = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        self.aspectratio = self.videowidth / self.videoheight
                except:
                        self.aspectratio = 1920 / 1080
                        #gets the fps
                
                fps = return_data.Fps.return_video_fps(fr'{videopath}')
                self.fps = fps
                width,height = return_data.VideoName.return_video_resolution(videopath)
                #Create files
                return_data.ManageFiles.create_folder(f'{renderdir}/{videoName}_temp/')
                return_data.ManageFiles.create_folder(f'{renderdir}/{videoName}_temp/input_frames')
        
                
                if settings.Image_Type != '.webp':
                        ffmpeg_cmd =(f'"{thisdir}/bin/ffmpeg" -i "{videopath}" -q:v 1 "{renderdir}/{videoName}_temp/input_frames/%08d{self.settings.Image_Type}" -y ') 
                else:
                        ffmpeg_cmd =(f'"{thisdir}/bin/ffmpeg" -i "{videopath}" -c:v libwebp -q:v 100 "{renderdir}/{videoName}_temp/input_frames/%08d.webp" -y ') 
                global output 
                run_subprocess_with_realtime_output(thread,self,ffmpeg_cmd,True)

                if self.localFile == True or self.youtubeFile == False:
                        os.system(f'"{thisdir}/bin/ffmpeg" -i "{videopath}" -vn -c:a aac -b:a 320k "{renderdir}/{videoName}_temp/audio.m4a" -y') # do same here i think maybe
                else:
                        os.system(f'mv "{thisdir}/audio.m4a" "{renderdir}/{videoName}_temp/audio.m4a"')
                
                global interpolation_sessions
                self.input_frames = len(os.listdir(f'{self.render_folder}/{self.videoName}_temp/input_frames/'))
                global frame_count
                self.filecount = 0
                frame_count = self.input_frames * self.times # frame count of video multiplied by times 
                if self.settings.RenderType == 'Optimized (Incremental)':
                        global frame_increments_of_interpolation
                        if self.settings.FrameIncrementsMode == 'Manual':
                            frame_increments_of_interpolation = self.settings.FrameIncrements
                        elif self.settings.FrameIncrementsMode == 'Automatic':
                            resolution = VideoName.return_video_resolution(self.input_file)
                            try:
                                frame_increments_of_interpolation = int(100*int(self.settings.VRAM)/(round(int(resolution[0])/1000)))
                            except:
                                 frame_increments_of_interpolation = int(100*int(self.settings.VRAM))
                            frame_increments_of_interpolation = int(frame_increments_of_interpolation)
                            print(frame_increments_of_interpolation)
                        self.frame_increments_of_interpolation = frame_increments_of_interpolation
                        interpolation_sessions = ceildiv(frame_count,frame_increments_of_interpolation)
                        for i in range(interpolation_sessions ):
                               os.mkdir(f'{renderdir}/{videoName}_temp/input_frames/{i}')
                               inc=0
                               files = os.listdir(f'{renderdir}/{videoName}_temp/input_frames/')
                               files.sort()
                               for j in files:
                                        if settings.Image_Type in j:
                                               if inc < frame_increments_of_interpolation/self.times:
                                                       os.rename(f'{renderdir}/{videoName}_temp/input_frames/{j}',f'{renderdir}/{videoName}_temp/input_frames/{i}/{str(inc).zfill(8)}{self.settings.Image_Type}')
                                                       inc+=1
                                               else:
                                                       break
                return_data.ManageFiles.create_folder(f'{renderdir}/{videoName}_temp/output_frames') # this is at end due to check in progressbar to start, bad implementation should fix later....
                return_data.ManageFiles.create_folder(f'{renderdir}/{videoName}_temp/output_frames/0/')
        except Exception as e:
                traceback_info = traceback.format_exc()
                log(f'{e} {traceback_info}')
                self.showDialogBox(e)
                
def end(thread,self,renderdir,videoName,videopath,times,outputpath,videoQuality,encoder,mode='interpolation'):
        settings = Settings()
        try:
                if self.output_folder == '':
                        outputpath = settings.OutputDir
                else:
                        outputpath = self.output_folder
                if mode == 'interpolation':
                        if return_data.ManageFiles.isfile(f'{outputpath}/{videoName}_{int(fps*times)}fps.mp4') == True:
                                i=1
                                while return_data.ManageFiles.isfile(f'{outputpath}/{videoName}_{int(fps*times)}fps({i}).mp4') == True:
                                        i+=1
                                output_video_file = f'{outputpath}/{videoName}_{int(fps*times)}fps({i}).mp4' 

                        else:
                                output_video_file = f'{outputpath}/{videoName}_{int(fps*times)}fps.mp4' 
                self.resIncrease = int(self.ui.Rife_Times.currentText()[0])
                if mode == 'upscale': # add upscale/realesrgan resolution bump here
                        upscaled_res = f'{int(width*self.resIncrease)}x{int(height*self.resIncrease)}'
                        if return_data.ManageFiles.isfile(f'{outputpath}/{videoName}_{upscaled_res}.mp4') == True:
                                i=1
                                while return_data.ManageFiles.isfile(f'{outputpath}/{videoName}_{upscaled_res}({i}).mp4') == True:
                                        i+=1
                                output_video_file = f'{outputpath}/{videoName}_{upscaled_res}({i}).mp4' 

                        else:
                                output_video_file = f'{outputpath}/{videoName}_{upscaled_res}.mp4'
                if settings.RenderType == 'Optimized' and os.path.exists(f'{self.settings.RenderDir}/{self.videoName}_temp/output_frames/videos.txt'):
                        if os.path.isfile(f'{renderdir}/{videoName}_temp/audio.m4a'):
                                os.system(f'"{thisdir}/bin/ffmpeg" -f concat -safe 0 -i "{self.settings.RenderDir}/{self.videoName}_temp/output_frames/videos.txt" -i "{self.settings.RenderDir}/{self.videoName}_temp/audio.m4a" -c copy "{output_video_file}"')
                        else:
                        
                                os.system(f'"{thisdir}/bin/ffmpeg" -f concat -safe 0 -i "{self.settings.RenderDir}/{self.videoName}_temp/output_frames/videos.txt" -c copy "{output_video_file}"') 
                else:
                        if os.path.isfile(f'{renderdir}/{videoName}_temp/audio.m4a'):
                                ffmpeg_cmd = (f'"{thisdir}/bin/ffmpeg" -framerate {fps*times} -i "{renderdir}/{videoName}_temp/output_frames/0/%08d{self.settings.Image_Type}" -i "{renderdir}/{videoName}_temp/audio.m4a" -c:v libx{encoder} -crf {videoQuality} -c:a copy  -pix_fmt yuv420p "{output_video_file}" -y')
                        else:
                        
                                ffmpeg_cmd = (f'"{thisdir}/bin/ffmpeg" -framerate {fps*times} -i "{renderdir}/{videoName}_temp/output_frames/0/%08d{self.settings.Image_Type}"  -c:v libx{encoder} -crf {videoQuality} -c:a copy  -pix_fmt yuv420p "{output_video_file}" -y') 
                        run_subprocess_with_realtime_output(thread,self,ffmpeg_cmd)
                os.system(f'rm -rf "{renderdir}/{videoName}_temp/audio.m4a"')
                try:
                        os.remove(f'{thisdir}/{videoName}')
                except:
                        pass
                #os.system(f'rm -rf "{renderdir}/{videoName}_temp/"') 
                os.system(f'rm -rf "{thisdir}/{self.input_file}"')
                try:
                       for i in os.listdir(f'{thisdir}'):
                        if '.mp4' in i:
                            os.system(f'rm -rf "{thisdir}/{i}"')
                except Exception as e:
                       log(str(e))
                os.chdir(thisdir)
                self.input_file = ''
                self.file_drop_widget.show()
                return output_video_file
                
        except Exception as e:
                traceback_info = traceback.format_exc()
                log(f'{e} {traceback_info}')
                self.showDialogBox(e)
