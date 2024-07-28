import argparse
import os

from src.RenderVideo import Render


class HandleApplication:
    def __init__(self):
        self.args = self.handleArguments()
        self.checkArguments()
        ffmpegSettings = Render(
            #model settings
            inputFile=self.args.input,
            outputFile=self.args.output,
            interpolateModel=self.args.interpolateModel,
            interpolateFactor=self.args.interpolateFactor,
            upscaleModel=self.args.upscaleModel,
            #backend settings
            device="cuda",
            backend=self.args.backend,
            precision="float16" if self.args.half else "float32",
            # ffmpeg settings
            overwrite=self.args.overwrite,
            crf=self.args.crf,
            benchmark=self.args.benchmark,
            encoder=self.args.custom_encoder
        )


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
            required=True,
            type=str,
        )
        parser.add_argument(
            "-o",
            "--output",
            default=None,
            help="output video path or PIPE",
            required=True,
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
            default=2,
        )
        parser.add_argument(
            "-c",
            "--cpu",
            help="use only CPU for upscaling, instead of cuda. default=auto",
            action="store_true",
        )
        parser.add_argument(
            "-f",
            "--format",
            help="output image format (jpg/png/webp, auto=same as input, default=auto)",
        )
        parser.add_argument(
            "--half",
            help="half precision, only works with NVIDIA RTX 20 series and above.",
            action="store_true",
        )
        parser.add_argument(
            "--bfloat16",
            help="like half precision, but more intesive. This can be used with a wider range of models than half.",
            action="store_true",
        )

        parser.add_argument(
            "-e",
            "--export",
            help="Export PyTorch models to ONNX and NCNN. Options: (onnx/ncnn)",
            default=None,
            type=str,
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
