import src.return_data as return_data
import os
from src.settings import *
import src.runAI.transition_detection
from src.return_data import *
from src.messages import *
from src.discord_rpc import *
import os
from modules.commands import *
from cv2 import VideoCapture, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT
import modules.upscale as upscale
import src.thisdir
from src.log import *
#this file changes the GUI aspects of the AI
thisdir = src.thisdir.thisdir()
homedir = os.path.expanduser(r"~")
settings = Settings()
def modelOptions(self):
    log('Model: RealSR')
    self.times=1
    self.render='esrgan'
    self.ui.Rife_Model.clear()
    self.ui.FPSPreview.setText('RES:')
    self.ui.Rife_Model.addItem('DF2K')
    self.ui.Rife_Model.addItem('DF2K_JPEG')
    self.ui.EnsembleCheckBox.hide()
    self.ui.EnsembleCheckBox.hide()
    self.ui.ensembleHelpButton.hide()
    self.ui.Rife_Model.setCurrentIndex(0)
    try:
        self.ui.Rife_Model.currentIndexChanged.disconnect()
    except:
        pass
    try:
        self.ui.RifeStart.clicked.disconnect()
    except:
          pass
    self.ui.Rife_Model.currentIndexChanged.connect((self.greyOutRealSRTimes))
    self.greyOutRealSRTimes()
    self.ui.RifeStart.clicked.connect(lambda: upscale.start_upscale(self,'realsr-ncnn-vulkan'))
    self.ui.Rife_Times.clear()
    self.ui.denoiseLevelLabel.hide()
    self.ui.denoiseLevelSpinBox.hide()
    
    
    self.ui.Rife_Times.addItem('4X')
    self.ui.Rife_Times.setCurrentIndex(0)

