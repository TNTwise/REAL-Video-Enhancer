import torch
import torch.nn.functional as F
import math
import os
from .Util import (
    currentDirectory,
    printAndLog,
    errorAndLog,
    modelsDirectory,
    check_bfloat16_support,
)

torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)


class InterpolateRifeTorch:
    """InterpolateRifeTorch class for video interpolation using RIFE model in PyTorch.

    Args:
        interpolateModelPath (str): Path to the pre-trained interpolation model.
        interpolateArch (str, optional): Interpolation architecture to use. Defaults to "rife413".
        width (int, optional): Width of the input frames. Defaults to 1920.
        height (int, optional): Height of the input frames. Defaults to 1080.
        device (str, optional): Device to use for computation. Defaults to "default".
        dtype (str, optional): Data type to use for computation. Defaults to "auto".
        backend (str, optional): Backend to use for computation. Defaults to "pytorch".
        UHDMode (bool, optional): Flag to enable UHD mode. Defaults to False.
        ensemble (bool, optional): Flag to enable ensemble mode. Defaults to False.
        trt_workspace_size (int, optional): Workspace size for TensorRT optimization. Defaults to 0.
        trt_max_aux_streams (int | None, optional): Maximum auxiliary streams for TensorRT optimization. Defaults to None.
        trt_optimization_level (int, optional): Optimization level for TensorRT optimization. Defaults to 5.
        trt_cache_dir (str, optional): Directory to cache TensorRT engine files. Defaults to modelsDirectory().
        trt_debug (bool, optional): Flag to enable TensorRT debug mode. Defaults to False.

    Methods:
        process(img0, img1, timestep):
            Processes the input frames and returns the interpolated frame.

        tensor_to_frame(frame):
            Converts a tensor to a frame for rendering.

        frame_to_tensor(frame):
            Converts a frame to a tensor for processing.
    def __init__(self, interpolateModelPath, interpolateArch="rife413", width=1920, height=1080, device="default", dtype="auto", backend="pytorch", UHDMode=False, ensemble=False, trt_workspace_size=0, trt_max_aux_streams=None, trt_optimization_level=5, trt_cache_dir=modelsDirectory(), trt_debug=False):
        pass

        Processes the input frames and returns the interpolated frame.

        Args:
            img0 (torch.Tensor): First input frame.
            img1 (torch.Tensor): Second input frame.
            timestep (float): Timestep between the input frames.

        Returns:
            torch.Tensor: Interpolated frame.
        pass

    def tensor_to_frame(self, frame):
        Converts a tensor to a frame for rendering.

        Args:
            frame (torch.Tensor): Input tensor representing a frame.

        Returns:
            numpy.ndarray: Rendered frame.
        pass

    def frame_to_tensor(self, frame):
        Converts a frame to a tensor for processing.

        Args:
            frame (numpy.ndarray): Input frame.

        Returns:
            torch.Tensor: Tensor representing the frame.
        pass"""

    @torch.inference_mode()
    def __init__(
        self,
        interpolateModelPath: str,
        interpolateArch: str = "rife413",
        width: int = 1920,
        height: int = 1080,
        device: str = "default",
        dtype: str = "auto",
        backend: str = "pytorch",
        UHDMode: bool = False,
        ensemble: bool = False,
        # trt options
        trt_workspace_size: int = 0,
        trt_max_aux_streams: int | None = None,
        trt_optimization_level: int = 5,
        trt_cache_dir: str = modelsDirectory(),
        trt_debug: bool = False,
    ):
        if device == "default":
            if torch.cuda.is_available():
                device = torch.device(
                    "cuda", 0
                )  # 0 is the device index, may have to change later
            else:
                device = torch.device("cpu")
        else:
            decice = torch.device(device)

        printAndLog("Using device: " + str(device))

        self.interpolateModel = interpolateModelPath
        self.width = width
        self.height = height
        self.device = device
        self.dtype = self.handlePrecision(dtype)
        self.backend = backend
        # set up streams for async processing
        self.stream = torch.cuda.Stream()
        self.prepareStream = torch.cuda.Stream()
        scale = 1
        if UHDMode:
            scale = 0.5
        with torch.cuda.stream(self.prepareStream):
            state_dict = torch.load(
                interpolateModelPath, map_location=self.device, weights_only=True, mmap=True
            )

            tmp = max(32, int(32 / scale))
            self.pw = math.ceil(self.width / tmp) * tmp
            self.ph = math.ceil(self.height / tmp) * tmp
            self.padding = (0, self.pw - self.width, 0, self.ph - self.height)

            # detect what rife arch to use
            match interpolateArch:
                case "rife46":
                    from .InterpolateArchs.RIFE.rife46IFNET import IFNet

                    v1 = False
                case "rife47":
                    from .InterpolateArchs.RIFE.rife47IFNET import IFNet

                    v1 = False
                case "rife413":
                    from .InterpolateArchs.RIFE.rife413IFNET import IFNet

                    v1 = False
                case "rife420":
                    from .InterpolateArchs.RIFE.rife420IFNET import IFNet

                    v1 = False
                case "rife421":
                    from .InterpolateArchs.RIFE.rife421IFNET import IFNet

                    v1 = False
                case "rife422-lite":
                    from .InterpolateArchs.RIFE.rife422_liteIFNET import IFNet

                    v1 = False
                case _:
                    errorAndLog("Invalid Interpolation Arch")

            # if 4.6 v1
            if v1:
                self.tenFlow_div = torch.tensor(
                    [(self.pw - 1.0) / 2.0, (self.ph - 1.0) / 2.0],
                    dtype=self.dtype,
                    device=self.device,
                )
                tenHorizontal = (
                    torch.linspace(-1.0, 1.0, self.pw, dtype=self.dtype, device=self.device)
                    .view(1, 1, 1, self.pw)
                    .expand(-1, -1, self.ph, -1)
                ).to(dtype=self.dtype, device=self.device)
                tenVertical = (
                    torch.linspace(-1.0, 1.0, self.ph, dtype=self.dtype, device=self.device)
                    .view(1, 1, self.ph, 1)
                    .expand(-1, -1, -1, self.pw)
                ).to(dtype=self.dtype, device=self.device)
                self.backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1)

            else:
                # if v2
                h_mul = 2 / (self.pw - 1)
                v_mul = 2 / (self.ph - 1)
                self.tenFlow_div = (
                    torch.Tensor([h_mul, v_mul])
                    .to(device=self.device, dtype=self.dtype)
                    .reshape(1, 2, 1, 1)
                )

                self.backwarp_tenGrid = torch.cat(
                    (
                        (torch.arange(self.pw) * h_mul - 1)
                        .reshape(1, 1, 1, -1)
                        .expand(-1, -1, self.ph, -1),
                        (torch.arange(self.ph) * v_mul - 1)
                        .reshape(1, 1, -1, 1)
                        .expand(-1, -1, -1, self.pw),
                    ),
                    dim=1,
                ).to(device=self.device, dtype=self.dtype)

            self.flownet = IFNet(
                scale=scale,
                ensemble=ensemble,
                dtype=self.dtype,
                device=self.device,
                width=self.width,
                height=self.height,
                backwarp_tenGrid=self.backwarp_tenGrid,
                tenFlow_div=self.tenFlow_div,
            )

            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items() if "module." in k
            }
            self.flownet.load_state_dict(state_dict=state_dict, strict=False)
            self.flownet.eval().to(device=self.device, dtype=self.dtype)

            if self.backend == "tensorrt":
                import tensorrt
                import torch_tensorrt

                trt_engine_path = os.path.join(
                    os.path.realpath(trt_cache_dir),
                    (
                        f"{os.path.basename(self.interpolateModel)}"
                        + f"_{self.pw}x{self.ph}"
                        + f"_{'fp16' if self.dtype == torch.float16 else 'fp32'}"
                        + f"_scale-{scale}"
                        + f"_ensemble-{ensemble}"
                        + f"_{torch.cuda.get_device_name(self.device)}"
                        + f"_trt-{tensorrt.__version__}"
                        + (
                            f"_workspace-{trt_workspace_size}"
                            if trt_workspace_size > 0
                            else ""
                        )
                        + (
                            f"_aux-{trt_max_aux_streams}"
                            if trt_max_aux_streams is not None
                            else ""
                        )
                        + (
                            f"_level-{trt_optimization_level}"
                            if trt_optimization_level is not None
                            else ""
                        )
                        + ".ts"
                    ),
                )
                if not os.path.isfile(trt_engine_path):
                    inputs = [
                        torch.zeros(
                            (1, 3, self.ph, self.pw), dtype=self.dtype, device=device
                        ),
                        torch.zeros(
                            (1, 3, self.ph, self.pw), dtype=self.dtype, device=device
                        ),
                        torch.zeros(
                            (1, 1, self.ph, self.pw), dtype=self.dtype, device=device
                        ),
                    ]
                    self.flownet = torch_tensorrt.compile(
                        self.flownet,
                        ir="dynamo",
                        inputs=inputs,
                        enabled_precisions={self.dtype},
                        debug=trt_debug,
                        workspace_size=trt_workspace_size,
                        min_block_size=1,
                        max_aux_streams=trt_max_aux_streams,
                        optimization_level=trt_optimization_level,
                        device=device,
                    )

                    torch_tensorrt.save(self.flownet, trt_engine_path, inputs=inputs)

                self.flownet = torch.export.load(trt_engine_path).module()

    def handlePrecision(self, precision):
        if precision == "auto":
            return torch.float16 if check_bfloat16_support() else torch.float32
        if precision == "float32":
            return torch.float32
        if precision == "float16":
            return torch.float16

    @torch.inference_mode()
    def process(self, img0, img1, timestep):
        with torch.cuda.stream(self.stream):
            timestep = torch.full(
                (1, 1, self.ph, self.pw), timestep, dtype=self.dtype, device=self.device
            )

            output = self.flownet(img0, img1, timestep)
            output = self.tensor_to_frame(output)
        self.stream.synchronize()
        return output

    @torch.inference_mode()
    def tensor_to_frame(self, frame: torch.Tensor):
        """
        Takes in a 4d tensor, undoes padding, and converts to np array for rendering
        """

        return frame.byte().contiguous().cpu().numpy()

    @torch.inference_mode()
    def frame_to_tensor(self, frame) -> torch.Tensor:
        with torch.cuda.stream(self.prepareStream):
            frame = (
                torch.frombuffer(
                    frame,
                    dtype=torch.uint8,
                )
                .to(device=self.device, dtype=self.dtype, non_blocking=True)
                .reshape(self.height, self.width, 3)
                .permute(2, 0, 1)
                .unsqueeze(0)
                / 255.0
            )
            frame = F.pad(
                (frame),
                self.padding,
            )
        self.prepareStream.synchronize()
        return frame

    def enqueueV3(self, context, bindings, stream, input_shapes):
        # Use the non-default stream for TensorRT inference
        context.enqueueV3(bindings, self.stream.cuda_stream, input_shapes)
