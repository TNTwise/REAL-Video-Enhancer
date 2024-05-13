from src.programData.checks import *


def rife_checkboxes(self):
    rife_list = [
        (self.ui.rife, "rife"),
        (self.ui.rifeanime, "rife-anime"),
        (self.ui.rifehd, "rife-HD"),
        (self.ui.rifeuhd, "rife-UHD"),
        (self.ui.rife2, "rife-v2"),
        (self.ui.rife23, "rife-v2.3"),
        (self.ui.rife24, "rife-v2.4"),
        (self.ui.rife30, "rife-v3.0"),
        (self.ui.rife31, "rife-v3.1"),
        (self.ui.rife4, "rife-v4"),
        (self.ui.rife41, "rife-v4.1"),
        (self.ui.rife42, "rife-v4.2"),
        (self.ui.rife43, "rife-v4.3"),
        (self.ui.rife44, "rife-v4.4"),
        (self.ui.rife45, "rife-v4.5"),
        (self.ui.rife46, "rife-v4.6"),
        (self.ui.rife47, "rife-v4.7"),
        (self.ui.rife48, "rife-v4.8"),
        (self.ui.rife49, "rife-v4.9"),
        (self.ui.rife410, "rife-v4.10"),
        (self.ui.rife411, "rife-v4.11"),
        (self.ui.rife412, "rife-v4.12"),
        (self.ui.rife412lite, "rife-v4.12-lite"),
        (self.ui.rife413, "rife-v4.13"),
        (self.ui.rife413lite, "rife-v4.13-lite"),
        (self.ui.rife414, "rife-v4.14"),
        (self.ui.rife414lite, "rife-v4.14-lite"),
        (self.ui.rife415, "rife-v4.15"),
        (self.ui.rife416lite, "rife-v4.16-lite"),
    ]
    # new models
    return rife_list

   


def rife_pin_functions(self):
    self.ui.rife.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rifeanime.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rifehd.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rifeuhd.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rife2.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rife23.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rife24.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rife30.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rife31.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rife4.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rife41.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rife42.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rife43.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rife44.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rife45.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rife46.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rife47.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rife48.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rife49.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rife410.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rife411.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rife412.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rife412lite.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rife413.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rife413lite.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rife414.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rife415.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rife416lite.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rife414lite.stateChanged.connect(self.checkbox_state_changed)
    self.ui.rife414lite.setEnabled(True)


def rife_cuda_checkboxes(self):
    if os.path.exists(f"{thisdir}/models/rife-cuda/"):
        for i in os.listdir(f"{thisdir}/models/rife-cuda/"):
            if i == "rife46":
                self.ui.rife46CUDA.setChecked(True)

            if i == "rife413-lite":
                self.ui.rife413liteCUDA.setChecked(True)
            if i == "rife414":
                self.ui.rife414CUDA.setChecked(True)
            if i == "rife414-lite":
                self.ui.rife414liteCUDA.setChecked(True)
            if i == "rife415":
                self.ui.rife415CUDA.setChecked(True)
            if i == "rife416-lite":
                self.ui.rife416liteCUDA.setChecked(True)
