from PyQt5.QtGui import QIcon
def show_scene_change_help(self):
    self.showDialogBox('Scene change detection sensitivity is based on a scale from 1 - 9.\n1 is the most sensative, meaning it will detect the most frames as changes in a scene, while 9will detect the least.')

def show_on_no_output_files(self):
    self.showDialogBox('Output frames or Audio file does not exist. Did you accidently delete them?',True)

def no_input_file(self):
    self.showDialogBox("No input file selected.",True)

def encoder_help(self):
    self.showDialogBox(".h264 is more standardized, but has worse quality. (shorter render time) \n.h265 is less standardized, but retains more visual quality. (longer render time)")
def cannot_detect_vram(self):
    self.showDialogBox("Cannot detect vram amount, please set this value in settings for increased performance.",True)