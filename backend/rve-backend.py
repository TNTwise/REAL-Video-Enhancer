import argparse
import os
import sys

from src.RenderVideo import Render

from src.Util import (
    checkForPytorch,
    checkForNCNN,
    checkForTensorRT,
    check_bfloat16_support,
    checkForDirectML,
    checkForDirectMLHalfPrecisionSupport,
)


class HandleApplication:
    def __init__(self):
        self.args = self.handleArguments()
        if not self.args.list_backends:
            self.checkArguments()
            Render(
                # model settings
                inputFile=self.args.input,
                outputFile=self.args.output,
                interpolateModel=self.args.interpolateModel,
                interpolateFactor=self.args.interpolateFactor,
                rifeVersion="v1", # some guy was angy about rifev2 being here, so I changed it to v1
                upscaleModel=self.args.upscaleModel,
                tile_size=self.args.tilesize,
                # backend settings
                device="default",
                backend=self.args.backend,
                precision=self.args.precision,
                # ffmpeg settings
                overwrite=self.args.overwrite,
                crf=self.args.crf,
                benchmark=self.args.benchmark,
                encoder=self.args.custom_encoder,
                # misc settingss
                pausedFile=self.args.pausedFile,
                sceneDetectMethod=self.args.sceneDetectMethod,
                sceneDetectSensitivity=self.args.sceneDetectSensitivity,
                sharedMemoryID=self.args.shared_memory_id,
                trt_optimization_level=self.args.tensorrt_opt_profile,
            )
        else:
            half_prec_supp = False
            availableBackends = []
            printMSG = ""
            if checkForNCNN():
                availableBackends.append("ncnn")
                printMSG += f"NCNN Version: 20220729\n"
                from rife_ncnn_vulkan_python import Rife
            if checkForDirectML():
                availableBackends.append("directml")
                import onnxruntime as ort

                printMSG += f"ONNXruntime Version: {ort.__version__}\n"
                half_prec_supp = checkForDirectMLHalfPrecisionSupport()
            if checkForPytorch():
                import torch

                availableBackends.append("pytorch")
                printMSG += f"PyTorch Version: {torch.__version__}\n"
                half_prec_supp = check_bfloat16_support()
                if checkForTensorRT():
                    """
                    checks for tensorrt availability, and the current gpu works with it (if half precision is supported)
                    Trt 10 only supports RTX 20 series and up.
                    Half precision is only availaible on RTX 20 series and up
                    """
                    if half_prec_supp:
                        import tensorrt

                        availableBackends.append("tensorrt")
                        printMSG += f"TensorRT Version: {tensorrt.__version__}\n"
                    else:
                        printMSG += "ERROR: Cannot use tensorrt backend, as it is not supported on your current GPU"

            printMSG += f"Half precision support: {half_prec_supp}\n"
            print("Available Backends: " + str(availableBackends))
            print(printMSG)

    def handleArguments(self) -> argparse.ArgumentParser:
        """_summary_

        Args:
            args (_type_): _description_

        """
        parser = argparse.ArgumentParser(
            description="Upscale any image, with most torch models, using spandrel."
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
            "--upscaleModel",
            help="Direct path to upscaling model, will automatically upscale if model is valid.",
            type=str,
        )
        parser.add_argument(
            "--interpolateModel",
            help="Direct path to interpolation model, will automatically upscale if model is valid.\n(Downloadable Options: [rife46, rife47, rife415, rife418, rife420, rife422, rife422lite]))",
            type=str,
        )
        parser.add_argument(
            "--interpolateFactor",
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
            "--sceneDetectMethod",
            help="Scene change detection to avoid interpolating transitions. (options=mean, mean_segmented, none)\nMean segmented splits up an image, and if an arbitrary number of segments changes are detected within the segments, it will trigger a scene change. (lower sensativity thresholds are not recommended)",
            type=str,
            default="mean",
        )
        parser.add_argument(
            "--sceneDetectSensitivity",
            help="Scene change detection sensitivity, lower number means it has a higher chance of detecting scene changes, with risk of detecting too many.",
            type=float,
            default=2.0,
        )
        parser.add_argument(
            "--overwrite",
            help="Overwrite output video if it already exists.",
            action="store_true",
        )
        parser.add_argument(
            "--crf",
            help="Constant rate factor for videos, lower setting means higher quality.",
            default="18",
        )
        parser.add_argument(
            "--custom_encoder",
            help="custom encoder",
            default="-c:v libx264",
            type=str,
        )
        parser.add_argument(
            "--tilesize",
            help="upscale images in smaller chunks, default is the size of the input video",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--benchmark",
            help="Overwrite output video if it already exists.",
            action="store_true",
        )
        parser.add_argument(
            "--shared_memory_id",
            help="Memory ID to share preview ons",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--list_backends",
            help="list out available backends",
            action="store_true",
        )
        parser.add_argument(
            "--pausedFile",
            help="File to store paused state (True means paused, False means unpaused)",
            type=str,
            default=None,
        )

        return parser.parse_args()

    def fullModelPathandName(self):
        return os.path.join(self.args.modelPath, self.args.modelName)

    def checkArguments(self):
        if (
            os.path.isfile(self.args.output)
            and not self.args.overwrite
            and not self.args.benchmark
        ):
            raise os.error("Output file already exists!")


if __name__ == "__main__":
    HandleApplication()
