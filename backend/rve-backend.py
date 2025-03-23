import argparse
import sys
import os
from src.RenderVideo import Render

from src.utils.Util import (
    checkForPytorchCUDA,
    checkForPytorchROCM,
    checkForNCNN,
    checkForTensorRT,
    check_bfloat16_support,
    checkForDirectML,
    checkForDirectMLHalfPrecisionSupport,
    get_gpus_ncnn,
    get_gpus_torch,
)


class HandleApplication:
    def __init__(self):
        self.args = self.handleArguments()
        if not self.args.list_backends:
            if not self.batchProcessing():
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
        half_prec_supp = False
        availableBackends = []
        printMSG = ""

        if checkForTensorRT():
            """
            checks for tensorrt availability, and the current gpu works with it (if half precision is supported)
            Trt 10 only supports RTX 20 series and up.
            Half precision is only availaible on RTX 20 series and up
            """
            import torch

            half_prec_supp = check_bfloat16_support()
            if half_prec_supp:
                import tensorrt

                availableBackends.append("tensorrt")
                printMSG += f"TensorRT Version: {tensorrt.__version__}\n"
            else:
                printMSG += "ERROR: Cannot use tensorrt backend, as it is not supported on your current GPU"

        if checkForPytorchCUDA():
            import torch

            availableBackends.append("pytorch (cuda)")
            printMSG += f"PyTorch Version: {torch.__version__}\n"
            half_prec_supp = check_bfloat16_support()
            pyTorchGpus = get_gpus_torch()
            for i, gpu in enumerate(pyTorchGpus):
                printMSG += f"PyTorch GPU {i}: {gpu}\n"

        if checkForPytorchROCM():
            availableBackends.append("pytorch (rocm)")
            import torch

            printMSG += f"PyTorch Version: {torch.__version__}\n"
            half_prec_supp = check_bfloat16_support()
            pyTorchGpus = get_gpus_torch()
            for i, gpu in enumerate(pyTorchGpus):
                printMSG += f"PyTorch GPU {i}: {gpu}\n"

        if checkForNCNN():
            availableBackends.append("ncnn")
            ncnnGpus = get_gpus_ncnn()
            printMSG += f"NCNN Version: 20220729\n"
            from rife_ncnn_vulkan_python import Rife

            for i, gpu in enumerate(ncnnGpus):
                printMSG += f"NCNN GPU {i}: {gpu}\n"
        if checkForDirectML():
            availableBackends.append("directml")
            import onnxruntime as ort

            printMSG += f"ONNXruntime Version: {ort.__version__}\n"
            half_prec_supp = checkForDirectMLHalfPrecisionSupport()

        printMSG += f"Half precision support: {half_prec_supp}\n"
        print("Available Backends: " + str(availableBackends))
        print(printMSG)

    def renderVideo(self):
        self.checkArguments()
        Render(
            # model settings
            inputFile=self.args.input,
            outputFile=self.args.output,
            interpolateModel=self.args.interpolate_model,
            interpolateFactor=self.args.interpolate_factor,
            upscaleModel=self.args.upscale_model,
            tile_size=self.args.tilesize,
            # backend settings
            device="default",
            backend=self.args.backend,
            precision=self.args.precision,
            pytorch_gpu_id=self.args.pytorch_gpu_id,
            ncnn_gpu_id=self.args.ncnn_gpu_id,
            # ffmpeg settings
            overwrite=self.args.overwrite,
            crf=self.args.crf,
            video_encoder_preset=self.args.video_encoder_preset,
            audio_encoder_preset=self.args.audio_encoder_preset,
            audio_bitrate=self.args.audio_bitrate,
            subtitle_encoder_preset=self.args.subtitle_encoder_preset,
            benchmark=self.args.benchmark,
            custom_encoder=self.args.custom_encoder,
            border_detect=self.args.border_detect,
            hdr_mode=self.args.hdr_mode,
            pixelFormat=self.args.video_pixel_format,
            # misc settings
            pause_shared_memory_id=self.args.pause_shared_memory_id,
            sceneDetectMethod=self.args.scene_detect_method,
            sceneDetectSensitivity=self.args.scene_detect_threshold,
            sharedMemoryID=self.args.preview_shared_memory_id,
            trt_optimization_level=self.args.tensorrt_opt_profile,
            upscale_output_resolution=self.args.upscale_output_resolution,
            UHD_mode=self.args.UHD_mode,
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
            help="Direct path to upscaling model, will automatically upscale if model is valid.",
            type=str,
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
            "--scene_detect_method",
            help="Scene change detection to avoid interpolating transitions. (options=mean, mean_segmented, none)\nMean segmented splits up an image, and if an arbitrary number of segments changes are detected within the segments, it will trigger a scene change. (lower sensativity thresholds are not recommended)",
            type=str,
            default="pyscenedetect",
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
            help="encoder preset that sets default encoder settings useful for hardware encoders.",
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
            help="pixel format for output video",
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
            help="encoder preset that sets default encoder settings",
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
            help="list out available backends",
            action="store_true",
        )
        parser.add_argument(
            "--pause_shared_memory_id",
            help="File to store paused state (True means paused, False means unpaused)",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--upscale_output_resolution",
            help="Resolution of output video, this is helpful for 4x models when you only want 2x upscaling. Ex: (1920x1080)",
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


if __name__ == "__main__":
    HandleApplication()
