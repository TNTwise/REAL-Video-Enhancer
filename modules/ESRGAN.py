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
    log('Model: RealESRGAN')
    self.times=1
    self.render='esrgan'
    self.ui.Rife_Model.clear()
    self.ui.FPSPreview.setText('RES:')
    self.ui.Rife_Model.addItem('Animation')
    self.ui.Rife_Model.addItem('Default')
    self.ui.EnsembleCheckBox.hide()
    for i in os.listdir(f'{settings.ModelDir}realesrgan/models/'):
                        if (os.path.splitext(i)[1])[1:] == 'bin':
                            if i not in default_models():
                                self.ui.Rife_Model.addItem(i.replace('.bin',''))
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
    self.ui.RifeStart.clicked.connect(lambda: upscale.start_upscale(self,'realesrgan-ncnn-vulkan'))
    self.ui.Rife_Times.clear()
    self.ui.denoiseLevelLabel.hide()
    self.ui.denoiseLevelSpinBox.hide()
    
    self.ui.Rife_Times.addItem('2X')
    self.ui.Rife_Times.addItem('3X')
    self.ui.Rife_Times.addItem('4X')
    self.ui.Rife_Times.setCurrentIndex(0)


def default_models():
    return ['realesr-animevideov3-x2.bin','realesr-animevideov3-x3.bin','realesr-animevideov3-x4.bin','realesrgan-x4plus-anime.bin','realesrgan-x4plus.bin']