
from PyQt5.QtGui import QIcon
def show_scene_change_help(self):
    self.showDialogBox('1 is the most sensitive, detecting the most frame changes in a scene.\n9 is the least sensitive, detecting fewest frame changes in a scene.')

def show_on_no_output_files(self):
    self.showDialogBox('Output frames or Audio file does not exist. Did you accidently delete them?',True)

def no_input_file(self):
    self.showDialogBox("No input file selected.",True)

def encoder_help(self):
    self.showDialogBox(".h264 is more standardized, but has worse quality. (shorter render time) \n.h265 is less standardized, but retains more visual quality. (longer render time)")

def cannot_detect_vram(self):
    self.showDialogBox("Cannot detect vram amount, please set this value in settings for increased performance.\nIf you are on integrated graphics, consider leaving this value at 1.",True)

def no_downloaded_models(self):
    self.showDialogBox("No models selected, please select at least one model to download.",True)

def failed_download(self):
    self.showDialogBox("Failed to download dependencies, please check your connection and try again.",True)

def image_help(self):
    self.showDialogBox("Extraction and render image type.\nJPG (recommended) lossy, low file size.\nPNG (longer render time) lossless, high file size.\nWEBP (longest render time) lossless, low file size.")

def vram_help(self):
    self.showDialogBox("VRAM limit for the program: Adjust to reduce system load or fix upscaling issues.\nChanging this alters the amount of VRAM the program has optimizes towards.")

def quotes(self):
    self.showDialogBox("Filenames with Quotes are not supported, please rename the file and try again.")

def not_enough_storage(self,predicted_space,total_space):
    self.showQuestionBox(f"Insufficient space for render.\nConsider using a more compressed image format, lowering enhancement iterations, changing the render folder disk, or switching to the Optimized Render Type.\nTotal space predicted for render: {int(predicted_space)} GB.\nTotal space left on disk: {int(total_space)} GB.\nContinue?")

def render_help(self):
    self.showDialogBox("The classic method consumes a significant amount of storage due to its inability to clear rendered frames.\n\nOptimized involves dividing the rendering process into smaller segments, each containing a predetermined number of frames. These segments are then compressed and merged into individual video segments. This approach effectively conserves storage space while maintaining video quality.")

def frame_increments_help(self):
    self.showDialogBox(f"This is how many frames will be rendered before they are merged into a video. (OPTIMIZED PRESET ONLY)\nLower values can lead to more storage usage as merging the video may take longer than the next render to run.\nHigher values can lead to more storage usage as there are more frames rendered before merging.\n(You usually want this on Automatic)")

def not_a_video(self):
    self.showDialogBox("This file is not a valid video.")

def not_valid_link(self):
    self.showDialogBox("This link is invalid.")

def not_enough_vram(self):
    self.showDialogBox("Your GPU does not meet the minimum required VRAM amount.\nMinimum: 4GB\nRecommended: 8GB\n It is highly advised to use Rife-V4 models.")
def too_large_video(self):
    self.showDialogBox('WARNING: The video you have chosen exceeds the maximum resolution (3840x2160).\nThis could cause issues in the render!')
def restart_app(self):
    self.showDialogBox("Please restart app for changes to take effect.")