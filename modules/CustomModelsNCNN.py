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
from modules.interpolate import *

# this file changes the GUI aspects of the AI
thisdir = src.programData.thisdir.thisdir()
homedir = os.path.expanduser(r"~")


def modelOptions(self):
    settings = Settings()
    self.times = 1
    log("Model: CUSTOM NCNN")
    self.render = "esrgan"

    self.ui.FPSToSign.hide()
    self.ui.Rife_Model.clear()
    self.ui.Rife_Times.clear()
    self.ui.FPSPreview.setText("RES:")
    self.ui.ensembleHelpButton.hide()
    for i in os.listdir(f"{thisdir}/models/custom_models_ncnn/models/"):
        if "bin" in i:
            self.ui.Rife_Model.addItem(i.replace(".bin", ""))

    self.ui.EnsembleCheckBox.hide()

    try:
        self.ui.Rife_Model.currentIndexChanged.disconnect()
    except:
        pass
    self.ui.Rife_Model.currentIndexChanged.connect(self.greyOutRealSRTimes)
    self.greyOutRealSRTimes()
    self.ui.Rife_Times.setCurrentIndex(0)
    self.ui.denoiseLevelLabel.hide()
    self.ui.denoiseLevelSpinBox.hide()

    try:
        self.ui.RifeStart.clicked.disconnect()
    except:
        pass
    self.ui.FPSFrom.hide()
    self.ui.FPSTo.hide()
    self.ui.Rife_Model.setEnabled(True)
    # lambda: startRender(self.input_file,f'{outputpath}/{os.path.basename(self.input_file)}_{self.fps*self.times}fps.mp4',self.times)
    self.ui.RifeStart.clicked.connect(
        lambda: upscale.start_upscale(self, "custom-models-ncnn-python")
    )
