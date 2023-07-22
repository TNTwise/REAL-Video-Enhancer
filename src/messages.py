from PyQt5.QtGui import QIcon
def show_scene_change_help(self):
    self.showDialogBox('1 is the most sensitive, detecting the most frame changes in a scene.\n9 is the least sensitive, detecting fewer frame changes in a scene.')

def show_on_no_output_files(self):
    self.showDialogBox('Output frames or Audio file does not exist. Did you accidently delete them?',True)

def no_input_file(self):
    self.showDialogBox("No input file selected.",True)

def encoder_help(self):
    self.showDialogBox(".h264 is more standardized, but has worse quality. (shorter render time) \n.h265 is less standardized, but retains more visual quality. (longer render time)")

def cannot_detect_vram(self):
    self.showDialogBox("Cannot detect vram amount, please set this value in settings for increased performance.",True)

def no_downloaded_models(self):
    self.showDialogBox("No models selected, please select at least one model to download.",True)

def failed_download(self):
    self.showDialogBox("Failed to download dependencies, please check your connection and try again.",True)

def image_help(self):
    self.showDialogBox("Extraction and render image type.\nJPG (recommended) lossy, low file size.\nPNG (high quality) lossless, high file size.\nWEBP (longer render time) lossless, low file size.")

def vram_help(self):
    self.showDialogBox("VRAM limit for the program: Adjust to reduce system load or fix upscaling issues.\nChanging this alters the amount of VRAM the program has optimizes towards.")