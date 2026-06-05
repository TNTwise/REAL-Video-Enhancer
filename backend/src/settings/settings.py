import os
from src.dirs import CONFIG_PATH, DEFAULT_VIDEOS_PATH

SETTINGS_FILE = os.path.join(CONFIG_PATH, '/settings.txt')

class Settings:
    def __init__(self):

        """
        The default settings are set here, and are overwritten by the settings in the settings file if it exists and the legnth of the settings is the same as the default settings.
        The key is equal to the name of the widget of the setting in the settings tab.
        """

        self.default_settings = {
            "precision": "auto",
            "tensorrt_optimization_level": "3",
            "dynamic_tensorrt_engine": "False",
            "encoder": "libx264",
            "video_encoder_speed": "medium",
            "audio_encoder": "copy_audio",
            "subtitle_encoder": "copy_subtitle",
            "audio_bitrate": "192k",
            "preview_enabled": "True",
            "scene_change_detection_method": "sudo_scene_detect",
            "scene_change_detection_enabled": "True",
            "scene_change_detection_threshold": "3.5",
            "discord_rich_presence": "False",
            "video_quality": "High",
            "output_folder_location": DEFAULT_VIDEOS_PATH,
            "use_same_output_folder_as_input_file_enabled": "False",
            "last_input_folder_location": DEFAULT_VIDEOS_PATH,
            "uhd_mode": "True",
            "ncnn_gpu_id": "0",
            "pytorch_gpu_id": "0",
            "auto_border_cropping": "False",
            "video_container": "mkv",
            "video_pixel_format": "yuv420p",
            "pytorch_version": "2.9.0",
            "pytorch_backend": "CUDA",
            "auto_hdr_mode": "True",
            "use_custom_encoder_command": "False",
            "encoder_command": "",
        }
        self.allowed_settings = {
            "precision": ("auto", "float32", "float16"),
            "tensorrt_optimization_level": ("0", "1", "2", "3", "4", "5"),
            "dynamic_tensorrt_engine": ("True", "False"),
            "encoder": (
                "libx264",
                "libx265",
                "vp9",
                "av1",
                "prores",
                "ffv1",
                "utvideo",
                "x264_nvenc",
                "x265_nvenc",
                "av1_nvenc (40 series and up)",
            ),
            "video_encoder_speed": ("placebo", "slow", "medium", "fast", "fastest"),
            "audio_encoder": ("aac", "libmp3lame", "opus", "copy_audio"),
            "audio_bitrate": "ANY",
            "subtitle_encoder": ("copy_subtitle", "srt", "ass", "webvtt"),
            "preview_enabled": ("True", "False"),
            "scene_change_detection_method": (
                "mean",
                "mean_segmented",
                "pyscenedetect",
                "sudo_scene_detect",
            ),
            "scene_change_detection_enabled": ("True", "False"),
            "scene_change_detection_threshold": [
                str(num / 10) for num in range(1, 100)
            ],
            "discord_rich_presence": ("True", "False"),
            "video_quality": (
                "Low",
                "Medium",
                "High",
                "Very_High",
                "Ultra",
                "Lossless",
            ),
            "output_folder_location": "ANY",
            "use_same_output_folder_as_input_file_enabled": ("True", "False"),
            "last_input_folder_location": "ANY",
            "uhd_mode": ("True", "False"),
            "ncnn_gpu_id": "ANY",
            "pytorch_gpu_id": "ANY",
            "auto_border_cropping": ("True", "False"),
            "video_container": ("mkv", "mp4", "mov", "webm", "avi"),
            "video_pixel_format": "ANY",
            "pytorch_version": ("2.10.0", "2.9.0", "2.6.0"),
            "pytorch_backend": "ANY",
            "auto_hdr_mode": ("True", "False"),
            "use_custom_encoder_command": ("True", "False"),
            "encoder_command": "ANY",
        }
        self.settings = self.default_settings.copy()
        if not os.path.isfile(SETTINGS_FILE):
            self.write_default_settings()
        self.read_settings()
        # check if the settings file is corrupted
        if len(self.default_settings) != len(self.settings):
            self.write_default_settings()

    def read_settings(self):
        """
        Reads the settings from the 'settings.txt' file and stores them in the 'settings' dictionary.

        Returns:
            None
        """
        with open(SETTINGS_FILE, "r") as file:
            try:
                for line in file:
                    key, value = line.strip().split(",")
                    self.settings[key] = value
            except (
                ValueError
            ):  # writes and reads again if the settings file is corrupted
                self.write_default_settings()
                self.read_settings()
    
    def write_setting(self, setting: str, value: str):
        """
        Writes the specified setting with the given value to the settings dictionary.

        Parameters:
        - setting (str): The name of the setting to be written, this will be equal to the widget name in the settings tab if set correctly.
        - value (str): The value to be assigned to the setting.

        Returns:
        None
        """
        if not setting in self.default_settings:
            raise ValueError("Not a valid setting")
        self.settings[setting] = value
        self.write_out_current_settings()

    def write_default_settings(self):
        """
        Writes the default settings to the settings file if it doesn't exist.

        Parameters:
            None

        Returns:
            None
        """
        self.settings = self.default_settings.copy()
        self.write_out_current_settings()
    
    def get_setting_value(self, setting: str) -> str:
        self.read_settings()
        if not setting in self.default_settings:
            raise ValueError("Not a valid setting")
        return self.settings[setting]

    def get_allowed_options(self, setting: str) -> list[str]:
        if not setting in self.allowed_settings:
            raise ValueError("Not a valid setting")
        return self.allowed_settings[setting]

    def write_out_current_settings(self):
        """
        Writes the current settings to a file.

        Parameters:
            self (SettingsTab): The instance of the SettingsTab class.

        Returns:
            None
        """
        with open(SETTINGS_FILE, "w") as file:
            for key, value in self.settings.items():
                if key in self.default_settings:  # check if the key is valid
                    if (
                        value in self.allowed_settings[key]
                        or self.allowed_settings[key] == "ANY"
                    ):  # check if it is in the allowed settings dict
                        file.write(f"{key},{value}\n")
                else:
                    self.write_default_settings()
