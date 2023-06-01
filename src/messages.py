
def show_scene_change_help(self):
    self.showDialogBox('Scene change detection sensitivity is based on a scale from 0.1 - 0.9.\n0.1 is the most sensative, meaning it will detect the most frames as changes in a scene, while 0.9 will detect the least.')

def show_on_no_output_files(self):
    self.showDialogBox('Output frames or Audio file does not exist. Did you accidently delete them?')

def no_input_file(self):
    self.showDialogBox("No input file selected.")

def encoder_help(self):
    self.showDialogBox(".h264 is more standardized, but has worse quality. (shorter render time) \n.h265 is less standardized, but retains more visual quality. (longer render time)")