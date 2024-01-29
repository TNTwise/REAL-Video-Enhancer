from PyQt5.QtWidgets import QFileDialog
from threading import *
from src.programData.settings import *
from src.programData.return_data import *
ManageFiles.create_folder(f'{thisdir}/files/')
import src.runAI.workers as workers
#import src.get_models as get_models
from src.misc.messages import *
def addToQueue(self):
    self.queueFile = QFileDialog.getOpenFileName(self, 'Open File', f'{homedir}',"Video files (*.mp4);;All files (*.*)")[0]
    if self.queueFile != '':
        self.QueueList.append(self.queueFile)
        self.queueVideoName = VideoName.return_video_name(self.queueFile)
        self.ui.QueueListWidget.addItem(self.queueVideoName)
        self.ui.QueueListWidget.show()