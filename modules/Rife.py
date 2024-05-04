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
import modules.interpolate as interpolate

# this file changes the GUI aspects of the AI
thisdir = src.programData.thisdir.thisdir()
homedir = os.path.expanduser(r"~")


def modelOptions(self):
    log("Model: Rife")
    self.times = 2
    self.render = "rife"
    self.ui.Rife_Model.clear()
    self.ui.Rife_Times.clear()
    self.ui.FPSPreview.setText("FPS:")

    self.ui.Rife_Times.addItem("2X")
    self.ui.Rife_Times.addItem("4X")
    self.ui.Rife_Times.addItem("8X")
    try:
        self.ui.Rife_Model.currentIndexChanged.disconnect()
    except:
        pass
    self.ui.Rife_Model.currentIndexChanged.connect(self.greyOutRifeTimes)
    self.ui.Rife_Times.setCurrentIndex(0)
    self.ui.denoiseLevelLabel.hide()
    self.ui.denoiseLevelSpinBox.hide()
    try:
        self.ui.RifeStart.clicked.disconnect()
    except:
        pass
    self.ui.RifeStart.clicked.connect(
        lambda: interpolate.start_interpolation(self, "rife-ncnn-vulkan")
    )
    models2 = self.get_models_from_dir("rife")
    models = []
    for i in models2:
        if "ensemble" not in i:
            models.append(i)
    models.sort()
    self.ui.EnsembleCheckBox.show()
    if len(self.get_models_from_dir("rife")) > 0:
        self.ui.Rife_Model.addItems(models)
        model_list = []
        for i in range(self.ui.defaultRifeModel.count()):
            item_text = self.ui.defaultRifeModel.itemText(i)
            model_list.append(item_text)

        for i in models:
            if i not in model_list:
                log(f"added model {i}")

                self.ui.defaultRifeModel.addItem(i)
        if f"{self.settings.DefaultRifeModel}" in models:
            self.ui.Rife_Model.setCurrentText(f"{self.settings.DefaultRifeModel}")
        else:
            models = sorted(self.get_models_from_dir("rife"))
            model_list = []
            for model in models:
                if "ensemble" in model:
                    pass
                else:
                    model_list.append(model)

            self.settings.change_setting(f"DefaultRifeModel", f"{model_list[-1]}")

            self.ui.Rife_Model.setCurrentText(f"{self.settings.DefaultRifeModel}")

        self.greyOutRifeTimes()


def ensemble_models():
    return [
        "rife-v4-ensemble",
        "rife-v4.1-ensemble",
        "rife-v4.2-ensemble",
        "rife-v4.3-ensemble",
        "rife-v4.4-ensemble",
        "rife-v4.5-ensemble",
        "rife-v4.6-ensemble",
        "rife-v4.7-ensemble",
        "rife-v4.8-ensemble",
        "rife-v4.9-ensemble",
        "rife-v4.10-ensemble",
        "rife-v4.11-ensemble",
        "rife-v4.12-ensemble",
        "rife-v4.12-lite-ensemble",
        "rife-v4.13-ensemble",
        "rife-v4.13-lite-ensemble",
        "rife-v4.14-ensemble",
        "rife-v4.14-lite-ensemble",
        "rife-v4.15-ensemble",
        "rife-v4.16-lite-ensemble",
    ]


def default_models():
    return ensemble_models() + [
        "rife",
        "rife-anime",
        "rife-HD",
        "rife-UHD",
        "rife-v2",
        "rife-v2.3",
        "rife-v2.4",
        "rife-v3.0",
        "rife-v3.1",
        "rife-v4",
        "rife-v4.1",
        "rife-v4.2",
        "rife-v4.3",
        "rife-v4.4",
        "rife-v4.5",
        "rife-v4.6",
        "rife-v4.7",
        "rife-v4.8",
        "rife-v4.9",
        "rife-v4.10",
        "rife-v4.11",
        "rife-v4.12",
        "rife-v4.12-lite",
        "rife-v4.13",
        "rife-v4.13-lite",
        "rife-v4.14",
        "rife-v4.14-lite",
        "rife-v4.15",
        "rife-v4.16-lite",
    ]
