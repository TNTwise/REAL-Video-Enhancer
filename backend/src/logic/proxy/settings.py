import os

from src.dirs import CONFIG_PATH, DEFAULT_VIDEOS_PATH

SETTINGS_FILE = CONFIG_PATH / "settings.txt"


# TODO: Make this cleaner
class PersistentSettingsProxy:
    def __init__(self):
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
            "output_folder_location": DEFAULT_VIDEOS_PATH._str,
            "use_same_output_folder_as_input_file_enabled": "False",
            "last_input_folder_location": DEFAULT_VIDEOS_PATH._str,
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
        if len(self.default_settings) != len(self.settings):
            self.write_default_settings()

    # --- Properties ---
    # I dont like all this shit,
    @property
    def precision(self) -> str:
        return self.settings["precision"]

    @precision.setter
    def precision(self, value: str):
        self._validate_and_set("precision", value)

    @property
    def tensorrt_optimization_level(self) -> str:
        return self.settings["tensorrt_optimization_level"]

    @tensorrt_optimization_level.setter
    def tensorrt_optimization_level(self, value: str):
        self._validate_and_set("tensorrt_optimization_level", value)

    @property
    def dynamic_tensorrt_engine(self) -> str:
        return self.settings["dynamic_tensorrt_engine"]

    @dynamic_tensorrt_engine.setter
    def dynamic_tensorrt_engine(self, value: str):
        self._validate_and_set("dynamic_tensorrt_engine", value)

    @property
    def encoder(self) -> str:
        return self.settings["encoder"]

    @encoder.setter
    def encoder(self, value: str):
        self._validate_and_set("encoder", value)

    @property
    def video_encoder_speed(self) -> str:
        return self.settings["video_encoder_speed"]

    @video_encoder_speed.setter
    def video_encoder_speed(self, value: str):
        self._validate_and_set("video_encoder_speed", value)

    @property
    def audio_encoder(self) -> str:
        return self.settings["audio_encoder"]

    @audio_encoder.setter
    def audio_encoder(self, value: str):
        self._validate_and_set("audio_encoder", value)

    @property
    def subtitle_encoder(self) -> str:
        return self.settings["subtitle_encoder"]

    @subtitle_encoder.setter
    def subtitle_encoder(self, value: str):
        self._validate_and_set("subtitle_encoder", value)

    @property
    def audio_bitrate(self) -> str:
        return self.settings["audio_bitrate"]

    @audio_bitrate.setter
    def audio_bitrate(self, value: str):
        self._validate_and_set("audio_bitrate", value)

    @property
    def preview_enabled(self) -> str:
        return self.settings["preview_enabled"]

    @preview_enabled.setter
    def preview_enabled(self, value: str):
        self._validate_and_set("preview_enabled", value)

    @property
    def scene_change_detection_method(self) -> str:
        return self.settings["scene_change_detection_method"]

    @scene_change_detection_method.setter
    def scene_change_detection_method(self, value: str):
        self._validate_and_set("scene_change_detection_method", value)

    @property
    def scene_change_detection_enabled(self) -> str:
        return self.settings["scene_change_detection_enabled"]

    @scene_change_detection_enabled.setter
    def scene_change_detection_enabled(self, value: str):
        self._validate_and_set("scene_change_detection_enabled", value)

    @property
    def scene_change_detection_threshold(self) -> str:
        return self.settings["scene_change_detection_threshold"]

    @scene_change_detection_threshold.setter
    def scene_change_detection_threshold(self, value: str):
        self._validate_and_set("scene_change_detection_threshold", value)

    @property
    def discord_rich_presence(self) -> str:
        return self.settings["discord_rich_presence"]

    @discord_rich_presence.setter
    def discord_rich_presence(self, value: str):
        self._validate_and_set("discord_rich_presence", value)

    @property
    def video_quality(self) -> str:
        return self.settings["video_quality"]

    @video_quality.setter
    def video_quality(self, value: str):
        self._validate_and_set("video_quality", value)

    @property
    def output_folder_location(self) -> str:
        return self.settings["output_folder_location"]

    @output_folder_location.setter
    def output_folder_location(self, value: str):
        self._validate_and_set("output_folder_location", value)

    @property
    def use_same_output_folder_as_input_file_enabled(self) -> str:
        return self.settings["use_same_output_folder_as_input_file_enabled"]

    @use_same_output_folder_as_input_file_enabled.setter
    def use_same_output_folder_as_input_file_enabled(self, value: str):
        self._validate_and_set("use_same_output_folder_as_input_file_enabled", value)

    @property
    def last_input_folder_location(self) -> str:
        return self.settings["last_input_folder_location"]

    @last_input_folder_location.setter
    def last_input_folder_location(self, value: str):
        self._validate_and_set("last_input_folder_location", value)

    @property
    def uhd_mode(self) -> str:
        return self.settings["uhd_mode"]

    @uhd_mode.setter
    def uhd_mode(self, value: str):
        self._validate_and_set("uhd_mode", value)

    @property
    def ncnn_gpu_id(self) -> str:
        return self.settings["ncnn_gpu_id"]

    @ncnn_gpu_id.setter
    def ncnn_gpu_id(self, value: str):
        self._validate_and_set("ncnn_gpu_id", value)

    @property
    def pytorch_gpu_id(self) -> str:
        return self.settings["pytorch_gpu_id"]

    @pytorch_gpu_id.setter
    def pytorch_gpu_id(self, value: str):
        self._validate_and_set("pytorch_gpu_id", value)

    @property
    def auto_border_cropping(self) -> str:
        return self.settings["auto_border_cropping"]

    @auto_border_cropping.setter
    def auto_border_cropping(self, value: str):
        self._validate_and_set("auto_border_cropping", value)

    @property
    def video_container(self) -> str:
        return self.settings["video_container"]

    @video_container.setter
    def video_container(self, value: str):
        self._validate_and_set("video_container", value)

    @property
    def video_pixel_format(self) -> str:
        return self.settings["video_pixel_format"]

    @video_pixel_format.setter
    def video_pixel_format(self, value: str):
        self._validate_and_set("video_pixel_format", value)

    @property
    def pytorch_version(self) -> str:
        return self.settings["pytorch_version"]

    @pytorch_version.setter
    def pytorch_version(self, value: str):
        self._validate_and_set("pytorch_version", value)

    @property
    def pytorch_backend(self) -> str:
        return self.settings["pytorch_backend"]

    @pytorch_backend.setter
    def pytorch_backend(self, value: str):
        self._validate_and_set("pytorch_backend", value)

    @property
    def auto_hdr_mode(self) -> str:
        return self.settings["auto_hdr_mode"]

    @auto_hdr_mode.setter
    def auto_hdr_mode(self, value: str):
        self._validate_and_set("auto_hdr_mode", value)

    @property
    def use_custom_encoder_command(self) -> str:
        return self.settings["use_custom_encoder_command"]

    @use_custom_encoder_command.setter
    def use_custom_encoder_command(self, value: str):
        self._validate_and_set("use_custom_encoder_command", value)

    @property
    def encoder_command(self) -> str:
        return self.settings["encoder_command"]

    @encoder_command.setter
    def encoder_command(self, value: str):
        self._validate_and_set("encoder_command", value)

    # --- Private helpers ---

    def _validate_and_set(self, setting: str, value: str):
        """Validate against allowed settings and persist."""
        if setting not in self.default_settings:
            raise ValueError(f"Not a valid setting: {setting}")
        allowed = self.allowed_settings[setting]
        if allowed != "ANY" and value not in allowed:
            raise ValueError(
                f"Invalid value '{value}' for setting '{setting}'. Allowed: {allowed}"
            )
        self.settings[setting] = value
        self.write_out_current_settings()

    # --- File I/O (unchanged) ---

    def read_settings(self):
        with open(SETTINGS_FILE, "r") as file:
            try:
                for line in file:
                    key, value = line.strip().split(",")
                    self.settings[key] = value
            except ValueError:
                self.write_default_settings()
                self.read_settings()

    def write_default_settings(self):
        self.settings = self.default_settings.copy()
        self.write_out_current_settings()

    def write_out_current_settings(self):
        with open(SETTINGS_FILE, "w") as file:
            for key, value in self.settings.items():
                if key in self.default_settings:
                    if (
                        self.allowed_settings[key] == "ANY"
                        or value in self.allowed_settings[key]
                    ):
                        file.write(f"{key},{value}\n")
                else:
                    self.write_default_settings()

    def write_setting(self, setting: str, value: str):
        # TODO: this is shit, and should use custom exceptions
        if setting not in self.settings:
            raise Exception("Setting not in settings!")
        throw = True
        for setting in self.default_settings:
            if value in setting:
                throw = False
        if self.default_settings[setting] == "ANY":
            throw = False
        if throw:
            raise Exception("Value not allowed")

        self.settings[setting] = value
        self.write_default_settings()

    def get_setting(self, setting):
        if setting not in self.settings:
            raise Exception("Setting not in settings!")
        return self.settings[setting]

    def get_allowed_options(self, setting: str):
        # TODO: again, this is shit
        if setting not in self.default_settings:
            raise Exception("Setting not in default settings")
        return self.default_settings[setting]
