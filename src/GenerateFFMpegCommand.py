
class FFMpegCommand:
    def __init__(self,
                 video_encoder: str,
                 video_encoder_speed: str,
                 video_quality: str,
                 video_pixel_format: str,
                 audio_encoder: str,
                 audio_bitrate: str,
                 hdr_mode: bool,
                 color_space: str,
                 color_primaries: str,
                 color_transfer: str,):
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
            command += [
                "-colorspace",
                self._color_space,
            ]
            encoder_params += f":colormatrix={self._color_space}:"
        
        
        if len(encoder_params) > 3:
            encoder_params = encoder_params[1:-1].replace("::", ":") # remove leading and trailing colons

        match self._video_encoder:
            case "libx264":
                command +=["-c:v","libx264"]
                match self._video_quality:
                    case "Very_High":
                        command +=["-crf","15"]
                    case "High":
                        command +=["-crf","18"]
                    case "Medium":
                        command +=["-crf","23"]
                    case "Low":
                        command +=["-crf","28"]
                
                match self._video_encoder_speed:
                    case "placebo":
                        command +=["-preset","placebo"]
                    case "slow":
                        command +=["-preset","slow"]
                    case "medium":
                        command +=["-preset","medium"]
                    case "fast":
                        command +=["-preset","fast"]
                    case "fastest":
                        command +=["-preset","veryfast"]

                if self._hdr_mode:
                    command += ["-x264-params", f'"{encoder_params}"']
                
                
            case "libx265":
                command +=["-c:v","libx265"]
                match self._video_quality:
                    case "Very_High":
                        command +=["-crf","15"]
                    case "High":
                        command +=["-crf","18"]
                    case "Medium":
                        command +=["-crf","23"]
                    case "Low":
                        command +=["-crf","28"]

                match self._video_encoder_speed:
                    case "placebo":
                        command +=["-preset","placebo"]
                    case "slow":
                        command +=["-preset","slow"]
                    case "medium":
                        command +=["-preset","medium"]
                    case "fast":
                        command +=["-preset","fast"]
                    case "fastest":
                        command +=["-preset","veryfast"]
                        
                if self._hdr_mode:
                    command += ["-x265-params", f'"{encoder_params}"']
                
            case "vp9":
                command +=["-c:v","libvpx-vp9"]
                match self._video_quality:
                    case "Very_High":
                        command +=["-crf","15"]
                    case "High":
                        command +=["-crf","20"]
                    case "Medium":
                        command +=["-crf","30"]
                    case "Low":
                        command +=["-crf","40"]

                match self._video_encoder_speed:
                    case "placebo":
                        command +=["-preset","placebo"]
                    case "slow":
                        command +=["-preset","slow"]
                    case "medium":
                        command +=["-preset","medium"]
                    case "fast":
                        command +=["-preset","fast"]
                    case "fastest":
                        command +=["-preset","veryfast"]

            case "av1":
                command +=["-c:v","libsvtav1"]
                match self._video_quality:
                    case "Very_High":
                        command +=["-cq:v","15"]
                    case "High":
                        command +=["-cq:v","18"]
                    case "Medium":
                        command +=["-cq:v","23"]
                    case "Low":
                        command +=["-cq:v","28"]
                match self._video_encoder_speed:
                    case "placebo":
                        command +=["-preset","0"]
                    case "slow":
                        command +=["-preset","4"]
                    case "medium":
                        command +=["-preset","8"]
                    case "fast":
                        command +=["-preset","12"]
                    case "fastest":
                        command +=["-preset","13"]
                
                    
            case "ffv1":
                command +=["-c:v","ffv1"]
                match self._video_quality:
                    case "Very_High":
                        command +=["-level","3"]
                    case "High":
                        command +=["-level","4"]
                    case "Medium":
                        command +=["-level","5"]
                    case "Low":
                        command +=["-level","6"]
            case "prores":
                command +=["-c:v","prores_ks"]
                match self._video_quality:
                    case "Very_High":
                        command +=["-profile:v","3"]
                    case "High":
                        command +=["-profile:v","2"]
                    case "Medium":
                        command +=["-profile:v","1"]
                    case "Low":
                        command +=["-profile:v","0"]
            case "x264_vulkan":
                command +=['-init_hw_device', 'vulkan=vkdev:0', '-filter_hw_device', 'vkdev', '-filter:v', f'format={self._video_pixel_format},hwupload']
                command +=["-c:v","h264_vulkan"]
                command +=["-quality","0"]
                """if self._hdr_mode:
                    command += ["-x264-params", f'"{encoder_params}"']"""
            case "x264_nvenc":
                command +=["-c:v","h264_nvenc"]
                match self._video_quality:
                    case "Very_High":
                        command +=["-cq:v","15"]
                    case "High":
                        command +=["-cq:v","18"]
                    case "Medium":
                        command +=["-cq:v","23"]
                    case "Low":
                        command +=["-cq:v","28"]

                """if self._hdr_mode:
                    command += ["-x264-params", f'"{encoder_params}"']"""
                match self._video_encoder_speed:
                    case "placebo":
                        command +=["-preset","p7"]
                    case "slow":
                        command +=["-preset","p6"]
                    case "medium":
                        command +=["-preset","p4"]
                    case "fast":
                        command +=["-preset","p2"]
                    case "fastest":
                        command +=["-preset","p1"]

            case "x265_nvenc":
                command +=["-c:v","hevc_nvenc"]
                match self._video_quality:
                    case "Very_High":
                        command +=["-cq:v","15"]
                    case "High":
                        command +=["-cq:v","18"]
                    case "Medium":
                        command +=["-cq:v","23"]
                    case "Low":
                        command +=["-cq:v","28"]
                if self._hdr_mode:
                    # HEVC-specific HDR parameters
                    command += ["-strict_gop", "1"]
                    command += ["-no-scenecut", "1"]
                    command += ["-spatial-aq", "1"]
                    command += ["-temporal-aq", "1"]
                    # HEVC can also handle HDR10 metadata
                    if self._color_transfer == "smpte2084":  # HDR10/PQ
                        command += ["-hdr10", "1"]
                        # HDR10 requires minimum 10-bit encoding
                        if not "10" in self._video_pixel_format:
                            print("Warning: HDR10 requires at least 10-bit color depth")
                """if self._hdr_mode:
                    command += ["-x265-params", f'"{encoder_params}"']"""
                match self._video_encoder_speed:
                    case "placebo":
                        command +=["-preset","p7"]
                    case "slow":
                        command +=["-preset","p6"]
                    case "medium":
                        command +=["-preset","p4"]
                    case "fast":
                        command +=["-preset","p2"]
                    case "fastest":
                        command +=["-preset","p1"]
            case "av1_nvenc":
                command +=["-c:v","av1_nvenc"]
                match self._video_quality:
                    case "Very_High":
                        command +=["-cq:v","15"]
                    case "High":
                        command +=["-cq:v","18"]
                    case "Medium":
                        command +=["-cq:v","23"]
                    case "Low":
                        command +=["-cq:v","28"]
                match self._video_encoder_speed:
                    case "placebo":
                        command +=["-preset","p7"]
                    case "slow":
                        command +=["-preset","p6"]
                    case "medium":
                        command +=["-preset","p4"]
                    case "fast":
                        command +=["-preset","p2"]
                    case "fastest":
                        command +=["-preset","p1"]
                
            case "h264_vaapi":
                command +=["-c:v","h264_vaapi"]
                match self._video_quality:
                    case "Very_High":
                        command +=["-crf","15"]
                    case "High":
                        command +=["-crf","18"]
                    case "Medium":
                        command +=["-crf","23"]
                    case "Low":
                        command +=["-crf","28"]
            case "h265_vaapi":
                command +=["-c:v","hevc_vaapi"]
                match self._video_quality:
                    case "Very_High":
                        command +=["-crf","15"]
                    case "High":
                        command +=["-crf","18"]
                    case "Medium":
                        command +=["-crf","23"]
                    case "Low":
                        command +=["-crf","28"]
            case "av1_vaapi":
                command +=["-c:v","av1_vaapi"]
                match self._video_quality:
                    case "Very_High":
                        command +=["-crf","15"]
                    case "High":
                        command +=["-crf","18"]
                    case "Medium":
                        command +=["-crf","23"]
                    case "Low":
                        command +=["-crf","28"]

            case _:
                command +=["-c:v","libx264"]
                match self._video_quality:
                    case "Very_High":
                        command +=["-crf","15"]
                    case "High":
                        command +=["-crf","18"]
                    case "Medium":
                        command +=["-crf","23"]
                    case "Low":
                        command +=["-crf","28"]
        
        command += ["-pix_fmt",self._video_pixel_format]

        match self._audio_encoder:
            case "copy_audio":
                command +=["-c:a","copy"]
            case "aac":
                command +=["-c:a","aac"]
            case "libmp3lame":
                command +=["-c:a","libmp3lame"]
            case "opus":
                command +=["-c:a","libopus"]
            case _:
                command +=["-c:a","copy"]
        
        if self._audio_encoder != "copy_audio":
            command += ["-b:a",self._audio_bitrate]

        return command

