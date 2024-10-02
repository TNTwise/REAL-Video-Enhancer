import os

from PySide6.QtWidgets import QMainWindow
from ..Util import currentDirectory, getPlatform, homedir, checkForWritePermissions
from .QTcustom import RegularQTPopup

class SettingsTab:
    def __init__(
        self,
        parent: QMainWindow,
        halfPrecisionSupport,
    ):
        self.parent = parent
        self.settings = Settings()

        self.connectWriteSettings()
        self.connectSettingText()

        # disable half option if its not supported
        if not halfPrecisionSupport:
            self.parent.precision.removeItem(1)
    

    """def connectWriteSettings(self):
        settings_and_combo_boxes = {
            "precision": self.parent.precision,
            "tensorrt_optimization_level": self.parent.tensorrt_optimization_level,
            "encoder": self.parent.encoder,
            
        }
        settings_and_check_boxes = {
            "preview_enabled": self.parent.preview_enabled,
            "scene_change_detection_enabled": self.parent.scene_change_detection_enabled,
            "discord_rich_presence": self.parent.discord_rich_presence,
        }
        for setting, combo_box in settings_and_combo_boxes.items():
            combo_box.currentIndexChanged.connect(
                lambda: self.settings.writeSetting(
                    setting, combo_box.currentText()
                )
            )
        for setting, check_box in settings_and_check_boxes.items():
            check_box.stateChanged.connect(
                lambda: self.settings.writeSetting(
                    setting, "True" if check_box.isChecked() else "False"
                )
            )
            print(setting)"""


    def connectWriteSettings(self):
        self.parent.precision.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "precision", self.parent.precision.currentText()
            )
        )
        self.parent.tensorrt_optimization_level.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "tensorrt_optimization_level",
                self.parent.tensorrt_optimization_level.currentText(),
            )
        )
        self.parent.encoder.currentIndexChanged.connect(
            lambda: self.settings.writeSetting(
                "encoder", self.parent.encoder.currentText()
            )
        )
        self.parent.preview_enabled.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "preview_enabled",
                "True" if self.parent.preview_enabled.isChecked() else "False",
            )
        )
        self.parent.scene_change_detection_enabled.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "scene_change_detection_enabled",
                "True"
                if self.parent.scene_change_detection_enabled.isChecked()
                else "False",
            )
        )
        self.parent.discord_rich_presence.stateChanged.connect(
            lambda: self.settings.writeSetting(
                "discord_rich_presence",
                "True" if self.parent.discord_rich_presence.isChecked() else "False",
            )
        )
        self.parent.scene_detection_threshold.valueChanged.connect(
            lambda: self.settings.writeSetting(
                "scene_detection_threshold",
                str(self.parent.scene_detection_threshold.value()),
            )
        )
        self.parent.scene_detection_threshold.valueChanged.connect(
            lambda: self.settings.writeSetting(
                "video_quality",
                str(self.parent.video_quality.currentText()),
            )
        )
        self.parent.output_folder_location.textChanged.connect(
            lambda:self.writeOutputFolder()
        )
        self.parent.resetSettingsBtn.clicked.connect(self.resetSettings)

    def writeOutputFolder(self):
        outputlocation = self.parent.output_folder_location.text()
        if checkForWritePermissions(outputlocation):
            self.settings.writeSetting(
                    "output_folder_location",
                    str(outputlocation),
                )
        else:
            RegularQTPopup("No permissions to export here!")

    def resetSettings(self):
        self.settings.writeDefaultSettings()
        self.settings.readSettings()
        self.connectSettingText()
        self.parent.switchToSettingsPage()

    def connectSettingText(self):
        if getPlatform() == "darwin":
            index = self.parent.encoder.findText("av1")
            self.parent.encoder.removeItem(index)

        self.parent.precision.setCurrentText(self.settings.settings["precision"])
        self.parent.tensorrt_optimization_level.setCurrentText(
            self.settings.settings["tensorrt_optimization_level"]
        )
        self.parent.encoder.setCurrentText(self.settings.settings["encoder"])
        self.parent.preview_enabled.setChecked(
            self.settings.settings["preview_enabled"] == "True"
        )
        self.parent.scene_change_detection_enabled.setChecked(
            self.settings.settings["scene_change_detection_enabled"] == "True"
        )
        self.parent.discord_rich_presence.setChecked(
            self.settings.settings["discord_rich_presence"] == "True"
        )
        self.parent.scene_detection_threshold.setValue(
            float(self.settings.settings["scene_detection_threshold"])
        )
        self.parent.video_quality.setCurrentText(
            self.settings.settings["video_quality"]
        )
        self.parent.output_folder_location.setText(
            self.settings.settings["output_folder_location"]
        )


class Settings:
    def __init__(self):
        self.settingsFile = os.path.join(currentDirectory(), "settings.txt")

        """
        The default settings are set here, and are overwritten by the settings in the settings file if it exists and the legnth of the settings is the same as the default settings.
        The key is equal to the name of the widget of the setting in the settings tab.
        """
        self.defaultSettings = {
            "precision": "auto",
            "tensorrt_optimization_level": "3",
            "encoder": "libx264",
            "preview_enabled": "True",
            "scene_change_detection_enabled": "True",
            "discord_rich_presence": "True",
            "scene_detection_threshold": "2.0",
            "video_quality": "High",
            "output_folder_location": os.path.join(f"{homedir}", "Videos") if getPlatform() != "darwin" else os.path.join(f"{homedir}", "Desktop")
        }
        self.allowedSettings = {
            "precision": ("auto", "float32", "float16"),
            "tensorrt_optimization_level": ("0", "1", "2", "3", "4", "5"),
            "encoder": ("libx264", "libx265", "vp9", "av1"),
            "preview_enabled": ("True", "False"),
            "scene_change_detection_enabled": ("True", "False"),
            "discord_rich_presence": ("True", "False"),
            "scene_detection_threshold": [str(num / 10) for num in range(1, 100)],
            "video_quality": ("Low", "Medium", "High", "Very High"),
            "output_folder_location": "ANY"
        }
        self.settings = self.defaultSettings.copy()
        if not os.path.isfile(self.settingsFile):
            self.writeDefaultSettings()
        self.readSettings()
        # check if the settings file is corrupted
        if len(self.defaultSettings) != len(self.settings):
            self.writeDefaultSettings()

    def readSettings(self):
        """
        Reads the settings from the 'settings.txt' file and stores them in the 'settings' dictionary.

        Returns:
            None
        """
        with open(self.settingsFile, "r") as file:
            try:
                for line in file:
                    key, value = line.strip().split(",")
                    self.settings[key] = value
            except (
                ValueError
            ):  # writes and reads again if the settings file is corrupted
                self.writeDefaultSettings()
                self.readSettings()

    def writeSetting(self, setting: str, value: str):
        """
        Writes the specified setting with the given value to the settings dictionary.

        Parameters:
        - setting (str): The name of the setting to be written, this will be equal to the widget name in the settings tab if set correctly.
        - value (str): The value to be assigned to the setting.

        Returns:
        None
        """
        self.settings[setting] = value
        self.writeOutCurrentSettings()

    def writeDefaultSettings(self):
        """
        Writes the default settings to the settings file if it doesn't exist.

        Parameters:
            None

        Returns:
            None
        """
        self.settings = self.defaultSettings.copy()
        self.writeOutCurrentSettings()

    def writeOutCurrentSettings(self):
        """
        Writes the current settings to a file.

        Parameters:
            self (SettingsTab): The instance of the SettingsTab class.

        Returns:
            None
        """
        with open(self.settingsFile, "w") as file:
            for key, value in self.settings.items():
                if key in self.defaultSettings:  # check if the key is valid
                    print(value)
                    if (
                        value in self.allowedSettings[key] or self.allowedSettings[key] == "ANY"
                    ):  # check if it is in the allowed settings dict
                        file.write(f"{key},{value}\n")
                else:
                    self.writeDefaultSettings()
