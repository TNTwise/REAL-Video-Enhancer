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
    self.times = 2
    log("Model: GMFSS")
    self.render = "rife"

    self.ui.FPSToSign.hide()
    self.ui.Rife_Model.clear()
    self.ui.Rife_Times.clear()
    self.ui.FPSPreview.setText("FPS:")
    self.ui.ensembleHelpButton.hide()
    self.ui.ImageExtractionCheckBox.hide()
    self.ui.Rife_Times.addItem("2X")
    self.ui.Rife_Times.addItem("3X")
    self.ui.Rife_Times.addItem("4X")
    self.ui.Rife_Times.addItem("5X")
    self.ui.Rife_Times.addItem("6X")
    self.ui.Rife_Times.addItem("7X")
    self.ui.Rife_Times.addItem("8X")
    model_list = ["gmfss-fortuna"]

    self.ui.Rife_Model.addItems(model_list)
    self.ui.Rife_Times.setEnabled(True)
    self.ui.EnsembleCheckBox.show()

    try:
        self.ui.Rife_Model.currentIndexChanged.disconnect()
    except:
        pass
    self.ui.Rife_Times.setCurrentIndex(0)
    self.ui.denoiseLevelLabel.hide()
    self.ui.denoiseLevelSpinBox.hide()

    try:
        self.ui.RifeStart.clicked.disconnect()
    except:
        pass
    self.ui.FPSFrom.hide()
    self.ui.FPSTo.hide()
    # lambda: startRender(self.input_file,f'{outputpath}/{os.path.basename(self.input_file)}_{self.fps*self.times}fps.mp4',self.times)
    self.ui.RifeStart.clicked.connect(lambda: start_interpolation(self, "gmfss-cuda"))
