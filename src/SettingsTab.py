import os

from PySide6.QtWidgets import QMainWindow
from .Util import currentDirectory

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
    
    def connectWriteSettings(self):
        self.parent.precision.currentIndexChanged.connect(lambda: self.settings.writeSetting("precision", self.parent.precision.currentText()))
        self.parent.tensorrt_optimization_level.currentIndexChanged.connect(lambda: self.settings.writeSetting("tensorrt_optimization_level", self.parent.tensorrt_optimization_level.currentText()))
        self.parent.encoder.currentIndexChanged.connect(lambda: self.settings.writeSetting("encoder", self.parent.encoder.currentText()))
    
    def connectSettingText(self):
        self.parent.precision.setCurrentText(self.settings.settings["precision"])
        self.parent.tensorrt_optimization_level.setCurrentText(self.settings.settings["tensorrt_optimization_level"])
        self.parent.encoder.setCurrentText(self.settings.settings["encoder"])

class Settings:
    def __init__(self):
        self.settingsFile = os.path.join(currentDirectory(),"settings.txt")
        
        """
        The default settings are set here, and are overwritten by the settings in the settings file if it exists and the legnth of the settings is the same as the default settings.
        The key is equal to the name of the widget of the setting in the settings tab.
        """
        self.defaultSettings={
                "precision":"auto",
                "tensorrt_optimization_level":"3",
                "encoder":"libx264",
            }
        self.allowedSettings = {
            "precision":("auto", "float32", "float16"),
            "tensorrt_optimization_level":("0", "1", "2", "3","4","5"),
            "encoder":("libx264", "libx265", "libvpx-vp9", "libaom-av1")
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
        with open('settings.txt', 'r') as file:
            try:
                for line in file:
                    key, value = line.strip().split(',')
                    self.settings[key] = value
            except ValueError: # writes and reads again if the settings file is corrupted
                self.writeDefaultSettings() 
                self.readSettings()
    def writeSetting(self, setting:str, value:str):
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
        self.settings=self.defaultSettings.copy()
        self.writeOutCurrentSettings()
    
    def writeOutCurrentSettings(self):
        """
        Writes the current settings to a file.

        Parameters:
            self (SettingsTab): The instance of the SettingsTab class.

        Returns:
            None
        """
        with open(self.settingsFile, 'w') as file:
            
            for key, value in self.settings.items():
                if key in self.defaultSettings: # check if the key is valid
                    if value in self.allowedSettings[key]: # check if it is in the allowed settings dict
                        file.write(f"{key},{value}\n")
                else:
                    self.writeDefaultSettings()
        