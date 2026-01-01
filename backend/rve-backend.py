import os
import argparse
import sys
from src.version import __version__
from src.utils.Util import log


class HandleApplication:
    def __init__(self):
        self.args = self.handleArguments()
        if self.args.version:
            print(f"{__version__}")
            sys.exit(0)
        
        if not self.args.list_backends:
            """from pyinstrument import Profiler
            profiler = Profiler()
            profiler.start()"""
            if self.args.ffmpeg_path == None:
                from src.utils.GetFFMpeg import download_ffmpeg
                self.ffmpeg_path = download_ffmpeg()
            else:
                self.ffmpeg_path = self.args.ffmpeg_path

            from src.utils.VideoInfo import OpenCVInfo, print_video_info
            
            if self.args.print_video_info:
                video_info = OpenCVInfo(self.args.print_video_info, ffmpeg_path=self.ffmpeg_path)
                print_video_info(video_info)
                #profiler.stop()
                #print(profiler.output_text(unicode=True, color=True))
                sys.exit(0)
            else:
                video_info = OpenCVInfo(self.args.input, ffmpeg_path=self.ffmpeg_path)
                print_video_info(video_info)
                
            

            self.checkArguments()

            

            if not self.batchProcessing():
                buffer_str = "=" * len(str(sys.argv[0]))
                log(buffer_str, False)
                log("RVE Backend Version: " + __version__, False)
                log(buffer_str, False)
                log("CLI Arguments: ", False)
                log(str(sys.argv), False)
                log(buffer_str, False)
                self.renderVideo()

        else:
            self.listBackends()

    def batchProcessing(self) -> bool:
        """
        Checks if the input is a text file. If so, it will start batch processing.
        """
        if os.path.splitext(self.args.input)[-1] == ".txt":
            with open(self.args.input, "r") as f:
                for line in f.readlines():  # iterate through each render
                    sys.argv[1:] = (
                        line.split()
                    )  # replace the line after the input file name
                    self.args = (
                        self.handleArguments()
                    )  # overwrite arguments based on the new sys.argv
                    self.renderVideo()
            return (
                True  # batch processing is being done, so no need to call renderVideo
            )
        else:
            return False

    def listBackends(self):
        from src.utils.BackendDetect import (
            BackendDetect
        )
        half_prec_supp = False
        availableBackends = []
        printMSG = "RVE Backend Version: " + __version__ + "\n"
        backendDetect = BackendDetect()

        tensorrt_ver = backendDetect.get_tensorrt()
        pytorch_device, pytorch_version = backendDetect.pytorch_device, backendDetect.pytorch_version
        ncnn_ver = backendDetect.get_ncnn()

        if tensorrt_ver:
            """
            checks for tensorrt availability, and the current gpu works with it (if half precision is supported)
            Trt 10 only supports RTX 20 series and up.
            Half precision is only availaible on RTX 20 series and up
            """

            half_prec_supp = backendDetect.get_half_precision()
            if half_prec_supp:
                availableBackends.append("tensorrt")
                printMSG += f"TensorRT Version: {tensorrt_ver}\n"
            else:
                printMSG += "ERROR: Cannot use tensorrt backend, as it is not supported on your current GPU"

        if pytorch_device:

            availableBackends.append(f"pytorch ({pytorch_device})")
            printMSG += f"PyTorch Version: {pytorch_version}\n"
            half_prec_supp = backendDetect.get_half_precision()
            pyTorchGpus = backendDetect.get_gpus_torch()
            for i, gpu in enumerate(pyTorchGpus):
                printMSG += f"PyTorch GPU {i}: {gpu}\n"

        if ncnn_ver:
            availableBackends.append("ncnn")
            ncnnGpus = backendDetect.get_gpus_ncnn()
            printMSG += f"NCNN Version: 20220729\n"
            from rife_ncnn_vulkan_python import Rife

            for i, gpu in enumerate(ncnnGpus):
                printMSG += f"NCNN GPU {i}: {gpu}\n"
       
        printMSG += f"Half precision support: {half_prec_supp}\n"
        printMSG += ("Available Backends: " + str(availableBackends))
        self.printMSG = printMSG
        print(printMSG)

    def renderVideo(self):
        
        from src.RenderVideo import Render
        

        Render(
            # model settings
            inputFile=self.args.input,
            outputFile=self.args.output,
            interpolateModel=self.args.interpolate_model,
            interpolateFactor=self.args.interpolate_factor,
            upscaleModel=self.args.upscale_model,
            extraRestorationModels=self.args.extra_restoration_models,
            sceneDetectModel=self.args.scene_detect_model,
            tile_size=self.args.tilesize,
            # backend settings
            device=self.args.device,
            backend=self.args.backend,
            precision=self.args.precision if self.args.device != "cpu" else "float32",
            pytorch_gpu_id=self.args.pytorch_gpu_id,
            ncnn_gpu_id=self.args.ncnn_gpu_id,
            cwd=self.args.cwd,
            # ffmpeg settings
            ffmpeg_path = self.ffmpeg_path,
            start_time=self.args.start_time,
            end_time=self.args.end_time,
            overwrite=self.args.overwrite,
            crf=self.args.crf,
            video_encoder_preset=self.args.video_encoder_preset,
            audio_encoder_preset=self.args.audio_encoder_preset,
            subtitle_encoder_preset=self.args.subtitle_encoder_preset,
            audio_bitrate=self.args.audio_bitrate,
            benchmark=self.args.benchmark,
            custom_encoder=self.args.custom_encoder,
            border_detect=self.args.border_detect,
            hdr_mode=self.args.hdr_mode,
            pixelFormat=self.args.video_pixel_format,
            merge_subtitles=self.args.merge_subtitles,
            # misc settings
            pause_shared_memory_id=self.args.pause_shared_memory_id,
            sceneDetectMethod=self.args.scene_detect_method,
            sceneDetectSensitivity=self.args.scene_detect_threshold,
            sharedMemoryID=self.args.preview_shared_memory_id,
            trt_optimization_level=self.args.tensorrt_opt_profile,
            trt_dynamic_shapes=self.args.tensorrt_dynamic_shapes,
            override_upscale_scale=self.args.override_upscale_scale,
            UHD_mode=self.args.UHD_mode,
            drba=False,
            slomo_mode=self.args.slomo_mode,
            dynamic_scaled_optical_flow=self.args.dynamic_scaled_optical_flow,
            ensemble=self.args.ensemble,
            output_to_mpv=self.args.output_to_mpv,
        )
        

    def handleArguments(self) -> argparse.ArgumentParser:
        """_summary_

        Args:
            args (_type_): _description_

        """
        parser = argparse.ArgumentParser(
            description="Backend to RVE, used to upscale and interpolate videos"
        )

        parser.add_argument(
            "-i",
            "--input",
            default=None,
            help="input video path",
            type=str,
        )
        parser.add_argument(
            "-o",
            "--output",
            default=None,
            help="output video path or PIPE",
            type=str,
        )
        parser.add_argument(
            "--start_time",
            default=None,
            help="Start of video to be rendered in seconds",
            type=float,
        )
        parser.add_argument(
            "--end_time",
            default=None,
            help="End of video to be rendered in seconds",
            type=float,
        )

        parser.add_argument(
            "--ffmpeg_path",
            default="./bin/ffmpeg",
            help="Path to the ffmpeg executable",
            type=str,
        )

        parser.add_argument(
            "-l",
            "--overlap",
            help="overlap size on tiled rendering (default=10)",
            default=0,
            type=int,
        )
        parser.add_argument(
            "-b",
            "--backend",
            help="backend used to upscale image. (pytorch/ncnn/tensorrt/directml, default=pytorch)",
            default="pytorch",
            type=str,
        )
        parser.add_argument(
            "--upscale_model",
            help="Direct path to upscaling model, will automatically upscale if model is valid. (arbitrary scale)",
            type=str,
        )
        parser.add_argument(
            "--extra_restoration_models",
            help="Direct path to a compression fixer model, will automatically inference if model are valid. (1x only) Can be parsed multiple times.",
            action='append',
        )
        parser.add_argument(
            "--interpolate_model",
            help="Direct path to interpolation model, will automatically interpolate if model is valid.\n(Downloadable Options: [rife46, rife47, rife415, rife418, rife420, rife422, rife422lite]))",
            type=str,
        )
        parser.add_argument(
            "--interpolate_factor",
            help="Multiplier for interpolation, will round up to nearest integer for interpolation but the fps will be correct",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--precision",
            help="sets precision for model, (auto/float16/float32, default=auto)",
            default="auto",
        )
        parser.add_argument(
            "--tensorrt_opt_profile",
            help="sets tensorrt optimization profile for model, (1/2/3/4/5, default=3)",
            type=int,
            default=3,
        )
        parser.add_argument(
            "--tensorrt_dynamic_shapes",
            help="Saves time by generating a dynamic engine so the tensorrt engine can be used with multiple resolutions.",
            action="store_true",
        )
        parser.add_argument(
            "--scene_detect_method",
            help="Scene change detection to avoid interpolating transitions. (options=mean, mean_segmented, none)\nMean segmented splits up an image, and if an arbitrary number of segments changes are detected within the segments, it will trigger a scene change. (lower sensativity thresholds are not recommended)",
            type=str,
            default="pyscenedetect",
        )
        parser.add_argument(
            "--scene_detect_model",
            help="Path to scene change model to use with model-based scene detection.",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--scene_detect_threshold",
            help="Scene change detection sensitivity, lower number means it has a higher chance of detecting scene changes, with risk of detecting too many.",
            type=float,
            default=4.0,
        )
        parser.add_argument(
            "--overwrite",
            help="Overwrite output video if it already exists.",
            action="store_true",
        )
        parser.add_argument(
            "--border_detect",
            help="Detects current borders and removes them, useful for removing black bars.",
            action="store_true",
        )
        parser.add_argument(
            "--crf",
            help="Constant rate factor for videos, lower setting means higher quality.",
            default="18",
        )
        parser.add_argument(
            "--video_encoder_preset",
            help="encoder preset that sets default encoder settings useful for hardware encoders. (Overwritten by custom encoder)",
            default="libx264",
            choices=[
                "libx264",
                "libx265",
                "vp9",
                "av1",
                "prores",
                "ffv1",
                "x264_vulkan",
                "x264_nvenc",
                "x265_nvenc",
                "av1_nvenc",
                "x264_vaapi",
                "x265_vaapi",
                "av1_vaapi",
            ],
            type=str,
        )
        parser.add_argument(
            "--video_pixel_format",
            help="pixel format for output video. (Overwritten by custom encoder)",
            default="yuv420p",
            choices=[
                "yuv420p",
                "yuv422p",
                "yuv444p",
                "yuv420p10le",
                "yuv422p10le",
                "yuv444p10le",
            ],
            type=str,
        )

        parser.add_argument(
            "--audio_encoder_preset",
            help="encoder preset that sets default encoder settings. (Overwritten by custom encoder)",
            default="copy_audio",
            choices=[
                "aac",
                "libmp3lame",
                "opus",
                "copy_audio",
            ],
            type=str,
        )
        parser.add_argument(
            "--subtitle_encoder_preset",
            help="encoder preset that sets default encoder settings",
            default="copy_subtitle",
            choices=[
                "srt",
                "ass",
                "webvtt",
                "copy_subtitle",
            ],
            type=str,
        )
        parser.add_argument(
            "--audio_bitrate",
            help="bitrate for audio if preset is used",
            default="192k",
            type=str,
        )

        parser.add_argument(
            "--custom_encoder",
            help="custom encoder",
            default=None,
            type=str,
        )
        parser.add_argument(
            "--tilesize",
            help="upscale images in smaller chunks, default is the size of the input video",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--device",
            help="Device used for inference. CUDA is used for any CUDA/ROCm device, MPS is for MacOS, and CPU is for well, cpu (cuda, mps, xpu, cpu - float32 only)",
            default="auto",
            choices=[
                "auto",
                "cuda",
                "mps",
                "xpu",
                "cpu",
            ]
        )
        parser.add_argument(
            "--pytorch_gpu_id",
            help="GPU ID for pytorch backend, default is 0",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--ncnn_gpu_id",
            help="GPU ID for ncnn backend, default is 0",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--benchmark",
            help="Benchmark without saving video",
            action="store_true",
        )
        parser.add_argument(
            "--UHD_mode",
            help="Lowers the resoltion flow is calculated at, speeding up model and saving vram. Helpful for higher resultions.",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--slomo_mode",
            help="Instead of increasing framerate, it will remain the same while just increasing the length of the video.",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--hdr_mode",
            help="Appends ffmpeg command to re encode with hdr colorspace",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--dynamic_scaled_optical_flow",
            help="Scale the optical flow based on the difference between frames, currently only works with the pytorch backend.",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--print_video_info",
            help="Print video information of the given video to this argument and exits.",
            default=None,
            type=str,
        )
        parser.add_argument(
            "--ensemble",
            help="Use ensemble when interpolating if the model supports it.",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--preview_shared_memory_id",
            help="Memory ID to share preview on",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--output_to_mpv",
            help="Outputs to mpv instead of an output file (requires mpv to be installed)",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--list_backends",
            help="list out available backends and exits",
            action="store_true",
        )
        parser.add_argument(
            "--version",
            help="prints backend version and exits",
            action="store_true",
        )
        parser.add_argument(
            "--pause_shared_memory_id",
            help="File to store paused state (True means paused, False means unpaused)",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--merge_subtitles",
            help="Merges subtitles into output video",
            action="store_true",
            default=True,
        )
        parser.add_argument(
            "--override_upscale_scale",
            help="Resolution of output video, this is helpful for 4x models when you only want 2x upscaling. Ex: (1920x1080)",
            type=int,
            default=None,
        )
        parser.add_argument(
            "--cwd",
            help="current working directory for the app",
            type=str,
            default=None,
        )
        # append extra args
        return parser.parse_args()

    def fullModelPathandName(self):
        return os.path.join(self.args.modelPath, self.args.modelName)

    def checkArguments(self):
        if (
            self.args.output is not None
            and os.path.isfile(self.args.output)
            and not self.args.overwrite
            and not self.args.benchmark
        ):
            raise os.error("Output file already exists!")
        if "http" not in self.args.input:
            if not os.path.isfile(self.args.input):
                raise os.error("Input file does not exist!")
        if self.args.tilesize < 0:
            raise ValueError("Tilesize must be greater than 0")
        if self.args.interpolate_factor < 0:
            raise ValueError("Interpolation factor must be greater than 0")
        if self.args.interpolate_factor == 1 and self.args.interpolate_model:
            raise ValueError(
                "Interpolation factor must be greater than 1 if interpolation model is used.\nPlease use --interpolateFactor 2 for 2x interpolation!"
            )
        if self.args.interpolate_factor != 1 and not self.args.interpolate_model:
            raise ValueError(
                "Interpolation factor must be 1 if no interpolation model is used.\nPlease use --interpolateFactor 1 for no interpolation!"
            )
        if self.args.backend == 'ncnn' and self.args.hdr_mode:
            print("WARNING: HDR mode is not supported with ncnn backend, falling back to SDR",file=sys.stderr)
            self.args.hdr_mode = False            

if __name__ == "__main__":
    
    HandleApplication()
    
