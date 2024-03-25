# This script creates a class that takes in params like "RealESRGAN or Rife", the model for the program,  the times of upscaling, and the path of the video, and the output path
# hz
import src.programData.return_data as return_data
import os
from src.programData.settings import *
import src.runAI.transition_detection
from src.programData.return_data import *
from src.misc.messages import *
from src.runAI.discord_rpc import *
import requests
import os
from cv2 import (
    VideoCapture,
    CAP_PROP_FRAME_WIDTH,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FPS,
    CAP_PROP_FRAME_COUNT,
)
from src.misc.log import *

# this file changes the GUI aspects of the AI
thisdir = src.programData.thisdir.thisdir()
from threading import Thread

try:
    from notify import *
    import notify
except Exception as e:
    log(f"ERROR: Importing of notifications failed! {e}")

import re

thisdir = src.programData.thisdir.thisdir()

import traceback

import time


def return_gpu_settings(self):
    if int(self.gpuMemory) < 1:
        gpu_usage = f"-j 1:1:1"
    else:
        num = int(int(self.gpuMemory))
        gpu_usage = f"-j {num}:{num}:{num}"
    return gpu_usage


def print_output(thread, self, extracting, pipe):
    total_frame_count = VideoName.return_video_frame_count(self.input_file)

    mode = "Merged"
    if extracting == True:
        mode = "Extracted"
        times = 1
    if mode == "Merged":
        times = self.times
    try:
        progressbar = "<"
        for i in range(pb_length):
            progressbar += "="
        progressbar += ">"
    except:
        pass

    while True:
        line = pipe.readline()
        if not line:
            break
        else:
            if "frame" in line:
                frame_num = re.findall(r"frame=[ ]*[\d]*", line)
                if len(frame_num) != 0:
                    thread.removelog.emit(f"Frames {mode}:")
                    frame_num = frame_num[0]

                    frame_num = frame_num.split("=")[1]
                    if mode == "Merged" and self.settings.RenderType == "Optimized":
                        pass
                    else:
                        thread.log.emit(
                            f"Frames {mode}: {frame_num} / {int(total_frame_count*times)}"
                        )
            if "[download]" in line:
                percent = re.findall(r"\[download\][ ]*[\d]*", line)
                percent = re.findall(r"[\d]*", percent[0])
                percent.reverse()
                percent = percent[:2]
                try:
                    percent = int(percent[1])

                    pb_value = int(pb_length * percent / 100)
                    try:
                        thread.removelog.emit(last_line)
                    except:
                        pass
                    last_line = (
                        f'{progressbar.replace("=","+", pb_value)} {str((percent))}%'
                    )
                    thread.log.emit(last_line)
                except:
                    pass
            line = line.replace(f"\n", "")
            log(line)


def run_subprocess_with_realtime_output(thread, self, command, extracting=False):
    self.ffmpeg = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
        bufsize=1,  # Line-buffered output
        universal_newlines=True,  # Ensure newline translation
    )

    stdout_thread = Thread(
        target=print_output,
        args=(
            thread,
            self,
            extracting,
            self.ffmpeg.stdout,
        ),
    )
    stderr_thread = Thread(
        target=print_output,
        args=(
            thread,
            self,
            extracting,
            self.ffmpeg.stderr,
        ),
    )

    stdout_thread.start()
    stderr_thread.start()

    # Wait for the process to finish
    self.ffmpeg.wait()

    # Wait for the output threads to finish printing
    stdout_thread.join()
    stderr_thread.join()
    log(self.ffmpeg.stdout)
    log(self.ffmpeg.stderr)

    return self.ffmpeg.returncode


def get_video_from_link(self, thread):
    global pb_length
    pb_length = 15
    if self.youtubeFile == True:
        thread.log.emit("[Downloading YouTube Video]")
        run_subprocess_with_realtime_output(
            thread, self, (f"{self.download_youtube_video_command}")
        )

    else:
        thread.log.emit("[Downloading Video]")
        response = requests.get(self.download_youtube_video_command, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))

        # Check if the download was successful
        if response.status_code != 200:
            raise Exception(
                f"Failed to download the file. Status code: {response.status_code}"
            )

        progressbar = "<"
        for i in range(pb_length):
            progressbar += "="
        progressbar += ">"
        with open(f"{thisdir}/{self.videoName}", "wb") as file:
            progress = 0
            for chunk in response.iter_content(chunk_size=8192):
                progress += len(chunk)

                pb_value = int(pb_length * progress / total_size_in_bytes)
                try:
                    thread.removelog.emit(last_line)
                except:
                    pass
                last_line = f'{progressbar.replace("=","+", pb_value)} {(str((progress/total_size_in_bytes)*100))[:5]}%'
                thread.log.emit(last_line)

                file.write(chunk)


def cudaAndNCNN(self, videopath, renderdir, videoName, thread):
    if self.localFile == False:
        get_video_from_link(self, thread)
    if "-ncnn-vulkan" in self.AI:
        self.ncnn = True
        self.cuda = False
    if "-cuda" in self.AI:
        self.ncnn = False
        self.cuda = True
        os.system(f'mkdir -p "{renderdir}/{videoName}_temp/output_frames/0/"')
    self.file_drop_widget.hide()
    global height
    global width
    width, height = return_data.VideoName.return_video_resolution(videopath)
    video = cv2.VideoCapture(self.input_file)
    try:
        self.videowidth = video.get(cv2.CAP_PROP_FRAME_WIDTH)

        self.videoheight = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.aspectratio = self.videowidth / self.videoheight
    except:
        self.aspectratio = 1920 / 1080

    if self.localFile == True or self.youtubeFile == False:
        thread.log.emit("[Extracting Audio]")
        os.system(
            f'"{thisdir}/bin/ffmpeg" -i "{videopath}" -vn -c:a aac -b:a 320k "{renderdir}/{videoName}_temp/audio.m4a" -y'
        )  # do same here i think maybe
    else:
        os.system(f'mv "{thisdir}/audio.m4a" "{renderdir}/{videoName}_temp/audio.m4a"')
    self.start_time = time.time()


def extractFramesAndAudio(
    thread, self, renderdir, videoName, videopath, times
):  # called by workers.py after a started thread, used by both upscaling and interpolation
    try:
        videoName = videoName.replace("'", "")
        videoName = videoName.replace('"', "")
        self.input_file = self.input_file.replace("'", "")
        self.input_file = self.input_file.replace('"', "")
        log(f"Starting Render, input_file={videopath}")
        settings = Settings()

        if check_for_write_permissions(settings.OutputDir) == False:
            self.showDialogBox(
                f"No write permissions!\n\nThis most likely means the output directory does not exist, in which create {homedir}/Videos, or you do not have permission to output there.\nEither set the output directory {homedir}/Videos or allow permission for the new directory."
            )

        # i need to clean this up lol
        os.system(f'rm -rf "{settings.RenderDir}/{videoName}_temp/"')
        # Gets the width and height
        global height
        global width

        

        # Calculate the aspect ratio
        self.videoName = videoName

        # gets the fps

        # Create files
        return_data.ManageFiles.create_folder(f"{renderdir}/{videoName}_temp/")
        return_data.ManageFiles.create_folder(
            f"{renderdir}/{videoName}_temp/input_frames"
        )

        cudaAndNCNN(self, videopath, renderdir, videoName, thread)
        self.fps = VideoName.return_video_framerate(f"{self.input_file}")
        if settings.Image_Type != ".webp":
            ffmpeg_cmd = f'"{thisdir}/bin/ffmpeg" -i "{videopath}" -q:v 1 -vf "scale=w={self.videowidth}:h={self.videoheight}" "{renderdir}/{videoName}_temp/input_frames/%08d{self.settings.Image_Type}" -y '
        else:
            ffmpeg_cmd = f'"{thisdir}/bin/ffmpeg" -i "{videopath}" -c:v libwebp -vf "scale=w={self.videowidth}:h={self.videoheight}" -q:v 100 "{renderdir}/{videoName}_temp/input_frames/%08d.webp" -y '
        global output
        log(run_subprocess_with_realtime_output(thread, self, ffmpeg_cmd, True))

        global interpolation_sessions
        self.input_frames = len(
            os.listdir(f"{settings.RenderDir}/{self.videoName}_temp/input_frames/")
        )
        global frame_count
        self.filecount = 0
        frame_count = (
            self.input_frames * self.times
        )  # frame count of video multiplied by times

        return_data.ManageFiles.create_folder(
            f"{renderdir}/{videoName}_temp/output_frames"
        )  # this is at end due to check in progressbar to start, bad implementation should fix later....
        return_data.ManageFiles.create_folder(
            f"{renderdir}/{videoName}_temp/output_frames/0/"
        )
        self.start_time = time.time()

        log(f"End of start function")
    except Exception as e:
        traceback_info = traceback.format_exc()
        log(f"{e} {traceback_info}")
        self.showDialogBox(str(f"{e}"))


def returnOutputFile(self, videoName, encoder):
    settings = Settings()
    if self.output_folder == "":
        outputpath = settings.OutputDir
    else:
        outputpath = self.output_folder
    if self.render == "rife":
        if (
            return_data.ManageFiles.isfile(
                f"{outputpath}/{videoName}_{round(self.fps*self.times)}fps.{return_data.returnContainer(encoder)}".replace(
                    "#", ""
                )
            )
            == True
        ):
            i = 1
            while (
                return_data.ManageFiles.isfile(
                    f"{outputpath}/{videoName}_{round(self.fps*self.times)}fps({i}).{return_data.returnContainer(encoder)}"
                )
                == True
            ):
                i += 1
            output_video_file = f"{outputpath}/{videoName}_{round(self.fps*self.times)}fps({i}).{return_data.returnContainer(encoder)}"

        else:
            output_video_file = f"{outputpath}/{videoName}_{round(self.fps*self.times)}fps.{return_data.returnContainer(encoder)}"
        self.resIncrease = int(self.ui.Rife_Times.currentText()[0])
    if self.render == "esrgan":  # add upscale/realesrgan resolution bump here
        upscaled_res = f"{int(width*self.resIncrease)}x{int(height*self.resIncrease)}"
        if (
            return_data.ManageFiles.isfile(
                f"{outputpath}/{videoName}_{upscaled_res}.{return_data.returnContainer(encoder)}"
            )
            == True
        ):
            i = 1
            while (
                return_data.ManageFiles.isfile(
                    f"{outputpath}/{videoName}_{upscaled_res}({i}).{return_data.returnContainer(encoder)}".replace(
                        "#", ""
                    )
                )
                == True
            ):
                i += 1
            output_video_file = f"{outputpath}/{videoName}_{upscaled_res}({i}).{return_data.returnContainer(encoder)}"

        else:
            output_video_file = f"{outputpath}/{videoName}_{upscaled_res}.{return_data.returnContainer(encoder)}"
        output_video_file = output_video_file.replace("#", "")

    return output_video_file


def end(
    thread,
    self,
    renderdir,
    videoName,
    videopath,
    times,
    outputpath,
    videoQuality,
    encoder,
    mode="interpolation",
):
    settings = Settings()
    try:
        log(f"Ending Render, input_file={videopath}")

        if self.output_folder == "":
            outputpath = settings.OutputDir
        else:
            outputpath = self.output_folder
        if mode == "interpolation":
            if (
                return_data.ManageFiles.isfile(
                    f"{outputpath}/{videoName}_{round(self.fps*times)}fps.{return_data.returnContainer(encoder)}".replace(
                        "#", ""
                    )
                )
                == True
            ):
                i = 1
                while (
                    return_data.ManageFiles.isfile(
                        f"{outputpath}/{videoName}_{round(self.fps*times)}fps({i}).{return_data.returnContainer(encoder)}"
                    )
                    == True
                ):
                    i += 1
                output_video_file = f"{outputpath}/{videoName}_{round(self.fps*times)}fps({i}).{return_data.returnContainer(encoder)}"

            else:
                output_video_file = f"{outputpath}/{videoName}_{round(self.fps*times)}fps.{return_data.returnContainer(encoder)}"
        self.resIncrease = int(self.ui.Rife_Times.currentText()[0])
        if mode == "upscale":  # add upscale/realesrgan resolution bump here
            upscaled_res = (
                f"{int(width*self.resIncrease)}x{int(height*self.resIncrease)}"
            )
            if (
                return_data.ManageFiles.isfile(
                    f"{outputpath}/{videoName}_{upscaled_res}.{return_data.returnContainer(encoder)}"
                )
                == True
            ):
                i = 1
                while (
                    return_data.ManageFiles.isfile(
                        f"{outputpath}/{videoName}_{upscaled_res}({i}).{return_data.returnContainer(encoder)}".replace(
                            "#", ""
                        )
                    )
                    == True
                ):
                    i += 1
                output_video_file = f"{outputpath}/{videoName}_{upscaled_res}({i}).{return_data.returnContainer(encoder)}"

            else:
                output_video_file = f"{outputpath}/{videoName}_{upscaled_res}.{return_data.returnContainer(encoder)}"
        output_video_file = output_video_file.replace("#", "")

        if "cuda" in self.AI:
            return output_video_file

        if settings.RenderType == "Optimized" and os.path.exists(
            f"{self.settings.RenderDir}/{self.videoName}_temp/output_frames/videos.txt"
        ):
            if os.path.isfile(f"{renderdir}/{videoName}_temp/audio.m4a"):
                ffmpeg_cmd = f'"{thisdir}/bin/ffmpeg" -f concat  -safe 0 -i "{self.settings.RenderDir}/{self.videoName}_temp/output_frames/videos.txt" -i "{self.settings.RenderDir}/{self.videoName}_temp/audio.m4a" -c copy  "{output_video_file}" -y'

            else:
                ffmpeg_cmd = f'"{thisdir}/bin/ffmpeg" -f concat -safe 0 -i "{self.settings.RenderDir}/{self.videoName}_temp/output_frames/videos.txt" -c copy "{output_video_file}" -y'
        else:
            if os.path.isfile(f"{renderdir}/{videoName}_temp/audio.m4a"):
                ffmpeg_cmd = f'"{thisdir}/bin/ffmpeg" -framerate {self.fps*times} -i "{renderdir}/{videoName}_temp/output_frames/0/%08d{self.settings.Image_Type}" -i "{renderdir}/{videoName}_temp/audio.m4a"  -c:v {return_data.returnCodec(self.settings.Encoder)} {returnCRFFactor(videoQuality,self.settings.Encoder)} -c:a copy  -pix_fmt yuv420p "{output_video_file}" -y'
            else:
                ffmpeg_cmd = f'"{thisdir}/bin/ffmpeg" -framerate {self.fps*times} -i "{renderdir}/{videoName}_temp/output_frames/0/%08d{self.settings.Image_Type}"  -c:v {return_data.returnCodec(encoder)} {returnCRFFactor(videoQuality,self.settings.Encoder)} -c:a copy  -pix_fmt yuv420p "{output_video_file}" -y'
        if run_subprocess_with_realtime_output(thread, self, ffmpeg_cmd) != 0:
            thread.log.emit(
                "ERROR: Couldn't output video! Maybe try changing the output directory or renaming the video to not contain quotes!"
            )
            os.system(f'rm -rf "{renderdir}/{videoName}_temp/"')
            os.system(f'rm -rf "{thisdir}/{self.input_file}"')
        else:
            os.system(f'rm -rf "{renderdir}/{videoName}_temp/audio.m4a"')
            try:
                os.remove(f"{thisdir}/{videoName}")

            except:
                pass
            os.system(f'rm -rf "{renderdir}/{videoName}_temp/"')
            os.system(f'rm -rf "{thisdir}/{self.input_file}"')
            try:
                for i in os.listdir(f"{thisdir}"):
                    if os.path.isfile(os.path.join(thisdir, i)):
                        if ".{return_data.returnContainer(encoder)}" in i:
                            os.system(f'rm -rf "{thisdir}/{i}"')

            except Exception as e:
                log(str(e))
            os.chdir(thisdir)
            self.output_file = output_video_file
            log(f"Finished Render, output_file={output_video_file}")

            if settings.Notifications == "Enabled":
                try:
                    notification(
                        "REAL Video Enhancer",
                        message="Render Finished",
                        app_name="REAL-Video-Enhancer",
                    )
                except Exception as e:
                    log(f"ERROR: Notification Failed! {e}")

            return output_video_file
        log(f"Failed Render, output_file=Null")
        return None
    except Exception as e:
        traceback_info = traceback.format_exc()
        log(f"{e} {traceback_info}")
        self.showDialogBox(e)
