import src.programData.return_data as return_data
import os
from src.programData.settings import *
import src.runAI.transition_detection
from src.programData.return_data import *
from src.misc.messages import *
from src.runAI.discord_rpc import *
import os
from modules.commands import *
from cv2 import (
    VideoCapture,
    CAP_PROP_FRAME_WIDTH,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FPS,
    CAP_PROP_FRAME_COUNT,
)
import modules.upscale as upscale
from src.misc.log import *

# this file changes the GUI aspects of the AI
thisdir = src.programData.thisdir.thisdir()
homedir = os.path.expanduser(r"~")
settings = Settings()


def modelOptions(self):
    log("Model: SPAN-ncnn")
    self.times = 1
    self.render = "esrgan"
    self.ui.Rife_Model.clear()
    self.ui.FPSPreview.setText("RES:")
    self.ui.Rife_Model.addItem("spanx2_ch52")
    self.ui.Rife_Model.addItem("spanx4_ch52")
    self.ui.EnsembleCheckBox.hide()
    self.ui.EnsembleCheckBox.hide()
    self.ui.FPSFrom.hide()
    self.ui.FPSToSign.hide()
    self.ui.FPSTo.hide()
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
    self.ui.RifeStart.clicked.connect(
        lambda: upscale.start_upscale(self, "span-ncnn-python")
    )
    self.ui.Rife_Times.clear()
    self.ui.denoiseLevelLabel.hide()
    self.ui.denoiseLevelSpinBox.hide()
    self.ui.Rife_Times.addItem("2X")
    self.ui.Rife_Times.addItem("4X")
    self.ui.Rife_Times.setCurrentIndex(0)
