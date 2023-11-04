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
#this file changes the GUI aspects of the AI
thisdir = src.thisdir.thisdir()
homedir = os.path.expanduser(r"~")
settings = Settings()
def modelOptions(self):
    self.times=1
    self.render='esrgan'
    self.ui.Rife_Model.clear()
    self.ui.FPSPreview.setText('RES:')
    self.ui.Rife_Model.addItem('Animation')
    self.ui.Rife_Model.addItem('Default')
    
    for i in os.listdir(f'{settings.ModelDir}realesrgan/models/'):
                        if (os.path.splitext(i)[1])[1:] == 'bin':
                            if i not in default_models():
                                self.ui.Rife_Model.addItem(i.replace('.bin',''))
    self.ui.Rife_Model.setCurrentIndex(0)
    try:
        self.ui.Rife_Model.currentIndexChanged.disconnect()
    except:
        pass
    self.ui.RifeStart.clicked.disconnect()
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

def image_options(self):
    self.times=1
    self.render='esrgan'
    self.ui.ModelCombo_Image.clear()
    self.ui.RESPreview.setText('RES:')
    self.ui.ModelCombo_Image.addItem('Animation')
    self.ui.ModelCombo_Image.addItem('Default')
    for i in os.listdir(f'{settings.ModelDir}realesrgan/models/'):
                        if (os.path.splitext(i)[1])[1:] == 'bin':
                            if i not in default_models():
                                self.ui.ModelCombo_Image.addItem(i.replace('.bin',''))
    self.ui.ModelCombo_Image.setCurrentIndex(0)
    try:
        self.ui.ModelCombo_Image.currentIndexChanged.disconnect()
    except:
        pass
    
    self.ui.ModelCombo_Image.currentIndexChanged.connect((self.greyOutRealSRTimes))
    self.greyOutRealSRTimes()
    self.ui.RifeStart.clicked.connect(lambda: upscale.start_upscale(self,'realesrgan-ncnn-vulkan'))
    self.ui.Times_Image.clear()
    self.ui.denoiseLevelLabel.hide()
    self.ui.denoiseLevelSpinBox.hide()
    
    self.ui.Times_Image.addItem('2X')
    self.ui.Times_Image.addItem('3X')
    self.ui.Times_Image.addItem('4X')
    self.ui.Times_Image.setCurrentIndex(0)

def default_models():
    return ['realesr-animevideov3-x2.bin','realesr-animevideov3-x3.bin','realesr-animevideov3-x4.bin','realesrgan-x4plus-anime.bin','realesrgan-x4plus.bin']