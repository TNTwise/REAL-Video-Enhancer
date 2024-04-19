import os
import subprocess
from src.programData.return_data import *
from src.programData.settings import *
from scenedetect import detect, ContentDetector
import cv2
import src.programData.thisdir
import math
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector


def generate_opposite_pair(number, start, end):
    if number < start or number > end:
        return None  # Number is outside the specified range

    opposite = end - (number - start)
    return opposite


class TransitionDetection:
    def __init__(self, originalSelf):
        self.settings = Settings()
        self.render_directory = self.settings.RenderDir
        self.input_file = originalSelf.input_file
        self.videoName = originalSelf.videoName

        self.thisdir = src.programData.thisdir.thisdir()
        self.fps = originalSelf.fps
        self.full_render_dir = f"{self.render_directory}/{self.videoName}_temp"
        self.times = originalSelf.times
        ManageFiles.create_folder(f"{self.full_render_dir}")
        ManageFiles.create_folder(f"{self.full_render_dir}/transitions")
        # Change scene\,0.6 to edit how much scene detections it does, do this for both ffmpeg commands

    def find_timestamps(self):
        """
        Find timestamps and save images of transitions to copy back into original frame folder

        This is written really poorly lmao, but it works :)
        """



        if self.settings.SceneChangeDetectionMode.lower() == "enabled":
            os.system(f'mkdir -p "{self.full_render_dir}/transitions/"')

            if self.settings.SceneChangeMethod == "ffmpeg":
                # This will get the timestamps of the scene changes, and for every scene change timestamp, i can times it by the fps count to get its current frame, and after interpolation, double it and replace it and it -1 frame with the transition frame stored in the transitions folder


                if (
                    self.settings.Image_Type == ".jpg"
                    or self.settings.Image_Type == ".png"
                ):
                    ffmpeg_cmd = [f"{thisdir}/bin/ffmpeg",
                                   "-i",
                                   f"{self.input_file}",
                                     f"-filter_complex",
                                     f"select=\'gt(scene\\,{self.settings.SceneChangeDetection})\',metadata=print",
                                     "-vsync",
                                     "vfr",
                                     "-q:v",
                                     "1",
                                     ]
                if self.settings.Image_Type == ".webp":
                    ffmpeg_cmd.append(f"{self.full_render_dir}/transitions/%07d.png")
                else:
                    ffmpeg_cmd.append(f"{self.full_render_dir}/transitions/%07d{self.settings.Image_Type}")

                output = subprocess.check_output(
                    ffmpeg_cmd, shell=False, stderr=subprocess.STDOUT
                )
                # Decode the output as UTF-8 and split it into lines
                output_lines = output.decode("utf-8").split("\n")
                # Create a list to store the timestamps
                timestamps = []

                # Iterate over the output lines and extract the timestamps
                for line in output_lines:
                    if "pts_time" in line:
                        timestamp = str(line.split("_")[3])
                        timestamp = str(timestamp.split(":")[1])
                        timestamps.append(
                            math.ceil(
                                round(float(timestamp) * float(self.fps))
                                * self.times
                            )
                        )
                self.timestamps = timestamps

            if self.settings.SceneChangeMethod == "pyscenedetect":
                settings = Settings()
                timestamps = []

                frame_nums = detect(f"{self.input_file}", ContentDetector())
                cap = cv2.VideoCapture(f"{self.input_file}")
                for i, scene in enumerate(frame_nums):
                    num = math.ceil(scene[0].get_frames() * self.times)
                    if num != 0:
                        timestamps.append(num)

                        cap.set(cv2.CAP_PROP_POS_FRAMES, scene[0].get_frames())
                        ret, frame = cap.read()
                        if ret:
                            if (
                                settings.Image_Type != ".webp"
                            ):
                                output_file = f"{self.full_render_dir}/transitions/{i:07d}{self.settings.Image_Type}"
                            else:
                                output_file = (
                                    f"{self.full_render_dir}/transitions/{i:07d}.png"
                                )
                            cv2.imwrite(output_file, frame)
                self.timestamps = timestamps
                log(timestamps)
                cap.release()
            return self.timestamps

    def get_frame_num(self, times, frames_subtracted=0):
        self.times = times
        settings = Settings()
        try:
            if self.settings.SceneChangeDetection != "Off":
                transitions = os.listdir(f"{self.full_render_dir}/transitions/")
                if not os.path.exists(f"{self.full_render_dir}/transitions/temp/"):
                    os.mkdir(f"{self.full_render_dir}/transitions/temp/")
                for iteration, i in enumerate(transitions):
                    if settings.Image_Type != ".webp":
                        # os.system(f'"{thisdir}/bin/ffmpeg" -i "{self.full_render_dir}/transitions/{str(str(iteration+1).zfill(7))}{settings.Image_Type}" -pix_fmt yuv444p "{self.full_render_dir}/transitions/temp/{self.timestamps[iteration]}{settings.Image_Type}"')
                        os.system(
                            f'mv "{self.full_render_dir}/transitions/{str(str(iteration+1).zfill(7))}{settings.Image_Type}" "{self.full_render_dir}/transitions/temp/{self.timestamps[iteration]}{settings.Image_Type}"'
                        )
                    else:
                        os.system(
                            f'mv "{self.full_render_dir}/transitions/{str(str(iteration+1).zfill(7))}.png" "{self.full_render_dir}/transitions/temp/{self.timestamps[iteration]}{settings.Image_Type}"'
                        )
                for i in self.timestamps:
                    if times > 1.9 and times < 2.1:
                        num_images = 2
                    elif times > 3.9 and times < 4.1:
                        num_images = 4
                    elif times > 7.9 and times < 8.1:
                        num_images = 8
                    else:
                        num_images = math.ceil(times) + 1
                    for j in range(num_images):
                        os.system(
                            f'cp "{self.full_render_dir}/transitions/temp/{i}{settings.Image_Type}" "{self.full_render_dir}/transitions/{str(int(i)-(j)+1).zfill(8)}{settings.Image_Type}"'
                        )
                os.system(f'rm -rf "{self.full_render_dir}/transitions/temp/"')
        except Exception as e:
            tb = traceback.format_exc()
            log(str(e) + str(tb))

    def merge_frames(self, iteration=None):
        if iteration == None:
            # os.system(f'cp "{self.full_render_dir}/transitions/"* "{self.full_render_dir}/output_frames/0/"')

            os.chdir(f"{self.full_render_dir}/transitions/")
            for i in os.listdir():
                if os.path.isfile(f"{self.full_render_dir}/output_frames/0/{i}"):
                    os.system(f'cp {i} "{self.full_render_dir}/output_frames/0/"')

            os.chdir(f"{self.thisdir}/")
        else:
            for i in os.listdir(f"{self.full_render_dir}/transitions/"):
                if os.path.isfile(f"{self.full_render_dir}/output_frames/{i}.mp4"):
                    try:
                        os.removedirs(
                            f"{self.full_render_dir}/transitions/{i}"
                        )  # i think i can use this instead of rm
                    except:
                        os.system(f'rm -rf "{self.full_render_dir}/transitions/{i}"')

            os.system(
                f'cp -r "{self.full_render_dir}/transitions/{iteration}"* "{self.full_render_dir}/output_frames/0/"'
            )
