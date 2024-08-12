import argparse
import os
import sys

from src.RenderVideo import Render
from src.Util import (
    checkForPytorch,
    checkForNCNN,
    checkForTensorRT,
    check_bfloat16_support,
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
                interpolateArch=self.args.interpolateArch,
                upscaleModel=self.args.upscaleModel,
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
                sceneDetectMethod=self.args.sceneDetectMethod,
                sceneDetectSensitivity=self.args.sceneDetectSensitivity,
                sharedMemoryID=self.args.shared_memory_id,
            )
        else:
            availableBackends = []
            printMSG = ""
            if checkForNCNN():
                availableBackends.append("ncnn")
                printMSG += f"NCNN Version: 20220729\n"
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
            "-t",
            "--tilesize",
            help="tile size (default=0)",
            default=0,
            type=int,
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
            help="backend used to upscale image. (pytorch/ncnn/tensorrt, default=pytorch)",
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
            help="Direct path to interpolation model, will automatically upscale if model is valid.",
            type=str,
        )
        parser.add_argument(
            "--interpolateFactor",
            help="Multiplier for interpolation",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--interpolateArch",
            help="Arch used for interpolation when using PyTorch inference. (rife46,rife413,rife420,rife421)",
            type=str,
            default="rife413",
        )
        parser.add_argument(
            "--precision",
            help="sets precision for model, (auto/float16/float32, default=auto)",
            default="auto",
        )
        parser.add_argument(
            "--sceneDetectMethod",
            help="Scene change detection to avoid interpolating transitions. (options=pyscenedetect, ffmpeg, none)",
            type=str,
            default="pyscenedetect",
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
        return parser.parse_args()

    def fullModelPathandName(self):
        return os.path.join(self.args.modelPath, self.args.modelName)

    def checkArguments(self):
        if self.args.backend == "pytorch":
            try:
                import torch
                import torchvision
                import spandrel
            except ImportError as e:
                raise ImportError(f"Cannot use PyTorch as the backend! {e}")

        if self.args.backend == "tensorrt":
            try:
                import torch
                import torchvision
                import spandrel
                import tensorrt
                import torch_tensorrt
            except ImportError as e:
                raise ImportError(f"Cannot use TensorRT as the backend! {e}")

        if self.args.backend == "ncnn":
            try:
                import rife_ncnn_vulkan_python
                from upscale_ncnn_py import UPSCALE
            except ImportError as e:
                raise ImportError(f"Cannot use NCNN as the backend! {e}")

        if os.path.isfile(self.args.output) and not self.args.overwrite:
            raise os.error("Output file already exists!")


if __name__ == "__main__":
    HandleApplication()
