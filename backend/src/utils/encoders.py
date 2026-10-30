from abc import ABC
from dataclasses import dataclass
from typing import Optional

@dataclass
class Encoder(ABC):
    preset_tag: str
    preInputsettings: Optional[str]
    postInputSettings: Optional[str]


@dataclass
class VideoEncoder(Encoder):
    qualityControlMode: Optional[str] = "-crf"
    ...


@dataclass
class AudioEncoder(Encoder): ...


@dataclass
class SubtitleEncoder(Encoder): ...


# audio encoder options
class copyAudio(AudioEncoder):
    preset_tag = "copy_audio"
    preInputsettings = None
    postInputSettings = "-c:a copy"


class aac(AudioEncoder):
    preset_tag = "aac"
    preInputsettings = None
    postInputSettings = "-c:a aac"


class libmp3lame(AudioEncoder):
    preset_tag = "libmp3lame"
    preInputsettings = None
    postInputSettings = "-c:a libmp3lame"


class opus(AudioEncoder):
    preset_tag = "opus"
    preInputsettings = None
    postInputSettings = "-c:a libopus"


# subtitle encoder options
class copySubtitles(SubtitleEncoder):
    preset_tag = "copy_subtitle"
    preInputsettings = None
    postInputSettings = "-c:s copy"


class srt(SubtitleEncoder):
    preset_tag = "srt"
    preInputsettings = None
    postInputSettings = "-c:s srt"


class ass(SubtitleEncoder):
    preset_tag = "ass"
    preInputsettings = None
    postInputSettings = "-c:s ass"


class webvtt(SubtitleEncoder):
    preset_tag = "webvtt"
    preInputsettings = None
    postInputSettings = "-c:s webvtt"


class libx264(VideoEncoder):
    preset_tag = "libx264"
    preInputsettings = None
    postInputSettings = "-c:v libx264"


class libx265(VideoEncoder):
    preset_tag = "libx265"
    preInputsettings = None
    postInputSettings = "-c:v libx265"


class vp9(VideoEncoder):
    preset_tag = "vp9"
    preInputsettings = None
    postInputSettings = "-c:v libvpx-vp9"
    qualityControlMode: str = "-cq:v"


class av1(VideoEncoder):
    preset_tag = "av1"
    preInputsettings = None
    postInputSettings = "-c:v libsvtav1"

class ffv1(VideoEncoder):
    preset_tag = "ffv1"
    preInputsettings = None
    postInputSettings = "-c:v ffv1"

class prores(VideoEncoder):
    preset_tag = "prores"
    preInputsettings = None
    postInputSettings = "-c:v prores_ks"


class x264_vulkan(VideoEncoder):
    preset_tag = "x264_vulkan"
    preInputsettings = "-init_hw_device vulkan=vkdev:0 -filter_hw_device vkdev"
    postInputSettings = "-filter:v format=nv12,hwupload -c:v h264_vulkan"
    # qualityControlMode: str = "-quality" # this is not implemented very well, quality ranges from 0-4 with little difference, so quality changing is disabled.


class x264_nvenc(VideoEncoder):
    preset_tag = "x264_nvenc"
    preInputsettings = "-hwaccel cuda -hwaccel_output_format cuda"
    postInputSettings = "-c:v h264_nvenc"
    qualityControlMode: str = "-cq:v"


class x265_nvenc(VideoEncoder):
    preset_tag = "x265_nvenc"
    preInputsettings = "-hwaccel cuda -hwaccel_output_format cuda"
    postInputSettings = "-c:v hevc_nvenc"
    qualityControlMode: str = "-cq:v"


class av1_nvenc(VideoEncoder):
    preset_tag = "av1_nvenc"
    preInputsettings = "-hwaccel cuda -hwaccel_output_format cuda"
    postInputSettings = "-c:v av1_nvenc"
    qualityControlMode: str = "-cq:v"


class h264_vaapi(VideoEncoder):
    preset_tag = "x264_vaapi"
    preInputsettings = "-hwaccel vaapi -hwaccel_output_format vaapi"
    postInputSettings = "-rc_mode CQP -c:v h264_vaapi"
    qualityControlMode: str = "-qp"


class h265_vaapi(VideoEncoder):
    preset_tag = "x265_vaapi"
    preInputsettings = "-hwaccel vaapi -hwaccel_output_format vaapi"
    postInputSettings = "-rc_mode CQP -c:v hevc_vaapi"
    qualityControlMode: str = "-qp"


class av1_vaapi(VideoEncoder):
    preset_tag = "av1_vaapi"
    preInputsettings = "-hwaccel vaapi -hwaccel_output_format vaapi"
    postInputSettings = "-rc_mode CQP -c:v av1_vaapi"
    qualityControlMode: str = "-qp"


class EncoderSettings:
    def __init__(self, encoder_preset, type="video"):
        self.encoder_preset = encoder_preset
        self.type = type
        self.encoder: Encoder = self.getEncoder()

    def getEncoder(self) -> Encoder:
        match self.type:
            case "video":
                encoder_type = VideoEncoder
            case "audio":
                encoder_type = AudioEncoder
            case "subtitle":
                encoder_type = SubtitleEncoder
            case _:
                raise ValueError("Not a valid encoder type")

        for encoder in encoder_type.__subclasses__():
            if encoder.preset_tag == self.encoder_preset:
                return encoder
        raise ValueError("No implemented encoder: " + str(self.encoder_preset))

    def getPreInputSettings(self) -> str:
        return self.encoder.preInputsettings

    def getPostInputSettings(self) -> str:
        return self.encoder.postInputSettings

    def getQualityControlMode(self) -> Optional[str]:
        if self.type == "video":
            return self.encoder.qualityControlMode
        return None

    def getPresetTag(self) -> str:
        return self.encoder.preset_tag
