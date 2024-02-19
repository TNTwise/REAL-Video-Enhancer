import os
from time import sleep

def returnRenderFPS(self):
    fp1 = self.files_processed
    sleep(1)
    fp2 = self.files_processed

    return (fp2-fp1)

def runRenderFPSThread(self):
    while os.path.exists(f'{self.settings.RenderDir}/{self.videoName}_temp/output_frames/0/'):
        self.currentRenderFPS = returnRenderFPS(self)