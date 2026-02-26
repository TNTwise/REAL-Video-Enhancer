class FFMpegCommand:
    def __init__(
        self,
        video_encoder: str,
        video_encoder_speed: str,
        video_quality: str,
        video_pixel_format: str,
        audio_encoder: str,
        audio_bitrate: str,
        subtitle_encoder: str,
        hdr_mode: bool,
        color_space: str,
        color_primaries: str,
        color_transfer: str,
        output_fps: str,
        use_ffmpeg_reduce_framerate: str,
    ):
        self._video_encoder = video_encoder
        self._video_encoder_speed = video_encoder_speed
        self._video_quality = video_quality
        self._video_pixel_format = video_pixel_format
        self._audio_encoder = audio_encoder
        self._audio_bitrate = audio_bitrate
        self._hdr_mode = hdr_mode
        self._color_space = color_space
        self._color_primaries = color_primaries
        self._color_transfer = color_transfer
        self._subtitle_encoder = subtitle_encoder
        self._output_fps = output_fps
        self._use_ffmpeg_reduce_framerate = use_ffmpeg_reduce_framerate

    def _get_video_quality(
        self,
        quality: str,
        lossless_crf: int = 0,
        ultra_crf: int = 10,
        very_high_crf: int = 15,
        high_crf: int = 18,
        medium_crf: int = 23,
        low_crf: int = 28,
    ) -> int:
        match quality:
            case "Lossless":
                return lossless_crf
            case "Ultra":
                return ultra_crf
            case "Very_High":
                return very_high_crf
            case "High":
                return high_crf
            case "Medium":
                return medium_crf
            case "Low":
                return low_crf
            case _:
                return medium_crf

    def _get_video_preset(
        self,
        speed: str,
        placebo: str | int = "placebo",
        slow: str | int = "slow",
        medium: str | int = "medium",
        fast: str | int = "fast",
        fastest: str | int = "veryfast",
    ) -> list[str]:
        match speed:
            case "placebo":
                preset = placebo
            case "slow":
                preset = slow
            case "medium":
                preset = medium
            case "fast":
                preset = fast
            case "fastest":
                preset = fastest
            case _:
                preset = medium
        return ["-preset", str(preset)]

    def build_command(self):
        command = []
        encoder_params = ":hdr-opt=1:"
        if self._color_primaries is not None:
            command += [
                "-color_primaries",
                self._color_primaries,
            ]
            encoder_params += f":colorprim={self._color_primaries}:"
        if self._color_transfer is not None:
            command += [
                "-color_trc",
                self._color_transfer,
            ]
            encoder_params += f":transfer={self._color_transfer}:"
        if self._color_space is not None:
            # Note: -colorspace is not a valid output encoding option in FFmpeg
            # Color matrix is set via encoder params (colormatrix=) instead
            encoder_params += f":colormatrix={self._color_space}:"

        if len(encoder_params) > 3:
            encoder_params = encoder_params[1:-1].replace(
                "::", ":"
            )  # remove leading and trailing colons

        match self._video_encoder:
            case "libx264":
                command += ["-c:v", "libx264"]
                command += ["-crf", str(self._get_video_quality(self._video_quality))]
                command += self._get_video_preset(self._video_encoder_speed)
                if self._hdr_mode:
                    command += ["-x264-params", f'"{encoder_params}"']
                if self._video_quality == "Lossless":
                    command += ["-qp", "0"]

            case "libx265":
                command += ["-c:v", "libx265"]
                command += ["-crf", str(self._get_video_quality(self._video_quality))]
                command += self._get_video_preset(self._video_encoder_speed)
                if self._hdr_mode:
                    if self._video_quality == "Lossless":
                        encoder_params += ":lossless=1"
                    command += ["-x265-params", f'"{encoder_params}"']
                elif self._video_quality == "Lossless":
                    command += ["-x265-params", "lossless=1"]

            case "vp9":
                command += ["-c:v", "libvpx-vp9"]
                command += [
                    "-crf",
                    str(
                        self._get_video_quality(
                            self._video_quality, high_crf=20, medium_crf=30, low_crf=40
                        )
                    ),
                ]
                command += self._get_video_preset(self._video_encoder_speed)

            case "av1":
                command += ["-c:v", "libsvtav1"]
                command += [
                    "-crf",
                    str(
                        self._get_video_quality(
                            self._video_quality,
                            ultra_crf=20,
                            very_high_crf=23,
                            high_crf=26,
                            medium_crf=30,
                            low_crf=35,
                        )
                    ),
                ]
                command += self._get_video_preset(
                    self._video_encoder_speed, 0, 4, 8, 12, 13
                )

            case "ffv1":
                command += ["-c:v", "ffv1"]

            case "utvideo":
                command += ["-c:v", "utvideo"]
                match self._video_quality:
                    case "Lossless":
                        command += ["-compression_level", "0"]
                    case "Ultra":
                        command += ["-compression_level", "1"]
                    case "Very_High":
                        command += ["-compression_level", "2"]
                    case "High":
                        command += ["-compression_level", "3"]
                    case "Medium":
                        command += ["-compression_level", "4"]
                    case "Low":
                        command += ["-compression_level", "5"]

            case "prores":
                command += ["-c:v", "prores_ks"]
                match self._video_quality:
                    case "Lossless":
                        command += ["-profile:v", "5"]
                    case "Ultra":
                        command += ["-profile:v", "4"]
                    case "Very_High":
                        command += ["-profile:v", "3"]
                    case "High":
                        command += ["-profile:v", "2"]
                    case "Medium":
                        command += ["-profile:v", "1"]
                    case "Low":
                        command += ["-profile:v", "0"]

            case "x264_vulkan":
                command += [
                    "-init_hw_device",
                    "vulkan=vkdev:0",
                    "-filter_hw_device",
                    "vkdev",
                    "-filter:v",
                    f"format={self._video_pixel_format},hwupload",
                ]
                command += ["-c:v", "h264_vulkan"]
                command += ["-quality", "0"]

            case "x264_nvenc":
                command += ["-c:v", "h264_nvenc"]
                command += ["-cq:v", str(self._get_video_quality(self._video_quality))]
                command += self._get_video_preset(
                    self._video_encoder_speed, "p7", "p6", "p4", "p2", "p1"
                )

            case "x265_nvenc":
                command += ["-c:v", "hevc_nvenc"]
                command += ["-cq:v", str(self._get_video_quality(self._video_quality))]
                command += self._get_video_preset(
                    self._video_encoder_speed, "p7", "p6", "p4", "p2", "p1"
                )
            case "av1_nvenc":
                command += ["-c:v", "av1_nvenc"]
                command += [
                    "-cq:v",
                    str(
                        self._get_video_quality(
                            self._video_quality,
                            ultra_crf=15,
                            very_high_crf=20,
                            high_crf=25,
                            medium_crf=30,
                            low_crf=35,
                        )
                    ),
                ]
                command += self._get_video_preset(
                    self._video_encoder_speed, "p7", "p6", "p4", "p2", "p1"
                )

            case "x264_vaapi":
                command += [
                    "-init_hw_device",
                    "vaapi=va:/dev/dri/renderD128",
                    "-filter_hw_device",
                    "vaapi",
                    "-filter:v",
                    f"format={self._video_pixel_format},hwupload",
                ]
                command += ["-c:v", "h264_vaapi"]
                command += ["-crf", str(self._get_video_quality(self._video_quality))]
            case "x265_vaapi":
                command += [
                    "-init_hw_device",
                    "vaapi=va:/dev/dri/renderD128",
                    "-filter_hw_device",
                    "vaapi",
                    "-filter:v",
                    f"format={self._video_pixel_format},hwupload",
                ]
                command += ["-c:v", "hevc_vaapi"]
                command += ["-crf", str(self._get_video_quality(self._video_quality))]
            case "av1_vaapi":
                command += ["-c:v", "av1_vaapi"]
                command += ["-crf", str(self._get_video_quality(self._video_quality))]

            case "h264_amf":
                command += ["-c:v", "h264_amf"]
                command += ["-crf", str(self._get_video_quality(self._video_quality))]
            case "h265_amf":
                command += ["-c:v", "hevc_amf"]
                command += ["-crf", str(self._get_video_quality(self._video_quality))]
            case "av1_amf":
                command += ["-c:v", "av1_amf"]
                command += ["-crf", str(self._get_video_quality(self._video_quality))]
            case _:
                command += ["-c:v", self._video_encoder]
                command += ["-crf", str(self._get_video_quality(self._video_quality))]

        command += ["-pix_fmt", self._video_pixel_format]

        match self._audio_encoder:
            case "copy_audio":
                command += ["-c:a", "copy"]
            case "aac":
                command += ["-c:a", "aac"]
            case "libmp3lame":
                command += ["-c:a", "libmp3lame"]
            case "opus":
                command += ["-c:a", "libopus"]
            case _:
                command += ["-c:a", "copy"]

        if self._audio_encoder != "copy_audio":
            command += ["-b:a", self._audio_bitrate]

        match self._subtitle_encoder:
            case "copy_subtitle":
                command += ["-c:s", "copy"]
            case "srt":
                command += ["-c:s", "srt"]
            case "ass":
                command += ["-c:s", "ass"]
            case "webvtt":
                command += ["-c:s", "webvtt"]
            case _:
                command += ["-c:s", "copy"]

        if self._use_ffmpeg_reduce_framerate:
            command += ["-vf", f"fps={self._output_fps}"]

        return command
