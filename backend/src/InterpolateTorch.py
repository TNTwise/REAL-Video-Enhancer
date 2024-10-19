import torch
import torch.nn.functional as F
from .InterpolateArchs.DetectInterpolateArch import ArchDetect
import math
import os
import logging
import gc
from .Util import (
    printAndLog,
    errorAndLog,
    modelsDirectory,
    check_bfloat16_support,
    log,
)
from time import sleep

torch.set_float32_matmul_precision("medium")
torch.set_grad_enabled(False)
logging.basicConfig(level=logging.INFO)


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
        ceilInterpolateFactor: int = 2,
        width: int = 1920,
        height: int = 1080,
        device: str = "default",
        dtype: str = "auto",
        backend: str = "pytorch",
        UHDMode: bool = False,
        ensemble: bool = False,
        rifeVersion: str = "v1",
        # trt options
        trt_workspace_size: int = 0,
        trt_max_aux_streams: int | None = None,
        trt_optimization_level: int = 5,
        trt_cache_dir: str = modelsDirectory(),
        trt_debug: bool = False,
        rife_trt_mode: str = "accurate",
        trt_static_shape: bool = True,
        trt_min_shape: list[int] = [128, 128],
        trt_opt_shape: list[int] = [1920, 1080],
        trt_max_shape: list[int] = [1920, 1080],
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
        self.trt_workspace_size = trt_workspace_size
        self.trt_max_aux_streams = trt_max_aux_streams
        self.trt_optimization_level = trt_optimization_level
        self.trt_cache_dir = trt_cache_dir
        self.backend = backend
        self.ceilInterpolateFactor = ceilInterpolateFactor
        # set up streams for async processing
        self.scale = 1
        self.img0 = None
        self.f0encode = None
        self.rife46 = False
        self.trt_debug = trt_debug # too much output, i would like a progress bar tho
        self.v1 = rifeVersion == "v1"
        self.rife_trt_mode = rife_trt_mode
        self.trt_static_shape = trt_static_shape
        self.trt_min_shape = trt_min_shape
        self.trt_opt_shape = trt_opt_shape
        self.trt_max_shape = trt_max_shape

        if UHDMode:
            self.scale = 0.5
        self._load()

    @torch.inference_mode()
    def _load(self):
        self.stream = torch.cuda.Stream()
        self.prepareStream = torch.cuda.Stream()
        with torch.cuda.stream(self.prepareStream):
            state_dict = torch.load(
                self.interpolateModel,
                map_location=self.device,
                weights_only=True,
                mmap=True,
            )
            # detect what rife arch to use
            ad = ArchDetect(self.interpolateModel)
            interpolateArch = ad.getArch()
            log("interp arch" + interpolateArch.lower())

            _pad = 32
            match interpolateArch.lower():
                case "rife46":
                    from .InterpolateArchs.RIFE.rife46IFNET import IFNet

                    self.rife46 = True
                case "rife47":
                    from .InterpolateArchs.RIFE.rife47IFNET import IFNet

                    num_ch_for_encode = 4
                    self.encode = torch.nn.Sequential(
                        torch.nn.Conv2d(3, 16, 3, 2, 1),
                        torch.nn.ConvTranspose2d(16, 4, 4, 2, 1),
                    )
                case "rife413":
                    from .InterpolateArchs.RIFE.rife413IFNET import IFNet, Head

                    num_ch_for_encode = 8
                    self.encode = Head()
                case "rife420":
                    from .InterpolateArchs.RIFE.rife420IFNET import IFNet, Head

                    num_ch_for_encode = 8
                    self.encode = Head()
                case "rife421":
                    from .InterpolateArchs.RIFE.rife421IFNET import IFNet, Head

                    num_ch_for_encode = 8
                    self.encode = Head()
                case "rife422lite":
                    from .InterpolateArchs.RIFE.rife422_liteIFNET import IFNet, Head

                    self.encode = Head()
                    num_ch_for_encode = 4
                case "rife425":
                    from .InterpolateArchs.RIFE.rife425IFNET import IFNet, Head

                    _pad = 64
                    num_ch_for_encode = 4
                    self.encode = Head()
                case _:
                    errorAndLog("Invalid Interpolation Arch")

            tmp = max(_pad, int(_pad / self.scale))
            self.pw = math.ceil(self.width / tmp) * tmp
            self.ph = math.ceil(self.height / tmp) * tmp
            self.padding = (0, self.pw - self.width, 0, self.ph - self.height)

            self.set_rife_args()  # sets backwarp_tenGrid and tenFlow_div
            # set up dynamic
            if self.trt_static_shape:
                self.dimensions = f"{self.pw}x{self.ph}"
            else:
                for i in range(2):
                    self.trt_min_shape[i] = math.ceil(self.trt_min_shape[i] / tmp) * tmp
                    self.trt_opt_shape[i] = math.ceil(self.trt_opt_shape[i] / tmp) * tmp
                    self.trt_max_shape[i] = math.ceil(self.trt_max_shape[i] / tmp) * tmp

                self.dimensions = (
                    f"min-{self.trt_min_shape[0]}x{self.trt_min_shape[1]}"
                    f"_opt-{self.trt_opt_shape[0]}x{self.trt_opt_shape[1]}"
                    f"_max-{self.trt_max_shape[0]}x{self.trt_max_shape[1]}"
                )

            # caching the timestep tensor in a dict with the timestep as a float for the key
            self.timestepDict = {}
            for n in range(self.ceilInterpolateFactor):
                timestep = n / (self.ceilInterpolateFactor)
                timestep_tens = torch.full(
                    (1, 1, self.ph, self.pw),
                    timestep,
                    dtype=self.dtype,
                    device=self.device,
                ).to(non_blocking=True)
                self.timestepDict[timestep] = timestep_tens

            self.flownet = IFNet(
                scale=self.scale,
                ensemble=False,
                dtype=self.dtype,
                device=self.device,
                width=self.width,
                height=self.height,
                rife_trt_mode=self.rife_trt_mode,
            )

            state_dict = {
                k.replace("module.", ""): v
                for k, v in state_dict.items()
                if "module." in k
            }
            head_state_dict = {
                k.replace("encode.", ""): v
                for k, v in state_dict.items()
                if "encode." in k
            }
            if not self.rife46:
                self.encode.load_state_dict(state_dict=head_state_dict, strict=True)
                self.encode.eval().to(device=self.device, dtype=self.dtype)
            self.flownet.load_state_dict(state_dict=state_dict, strict=False)
            self.flownet.eval().to(device=self.device, dtype=self.dtype)
            if self.backend == "tensorrt":
                import tensorrt
                import torch_tensorrt
                from .InterpolateArchs.RIFE.custom_warplayer import WarpPluginCreator

                registry = tensorrt.get_plugin_registry()
                registry.register_creator(WarpPluginCreator())
                base_trt_engine_path = os.path.join(
                    os.path.realpath(self.trt_cache_dir),
                    (
                        f"{os.path.basename(self.interpolateModel)}"
                        + f"_{self.dimensions}"
                        + f"_{'fp16' if self.dtype == torch.float16 else 'fp32'}"
                        + f"_scale-{self.scale}"
                        + f"_ensemble-False"
                        + f"_{torch.cuda.get_device_name(self.device)}"
                        + f"torch_tensorrt-{torch_tensorrt.__version__}"
                        + f"_trt-{tensorrt.__version__}"
                        + (f"rife_trt_mode-" + self.rife_trt_mode)
                        + (f"model_version-2")
                        + (
                            f"_workspace-{self.trt_workspace_size}"
                            if self.trt_workspace_size > 0
                            else ""
                        )
                        + (
                            f"_aux-{self.trt_max_aux_streams}"
                            if self.trt_max_aux_streams is not None
                            else ""
                        )
                        + (
                            f"_level-{self.trt_optimization_level}"
                            if self.trt_optimization_level is not None
                            else ""
                        )
                    ),
                )
                trt_engine_path = base_trt_engine_path + ".dyn"
                encode_trt_engine_path = base_trt_engine_path + "_encode.dyn"
                # lay out inputs

                # load flow engine
                if not os.path.isfile(trt_engine_path):
                    exampleInputs = [
                        torch.zeros(
                            [1, 3, self.ph, self.pw],
                            dtype=self.dtype,
                            device=self.device,
                        ),
                        torch.zeros(
                            [1, 3, self.ph, self.pw],
                            dtype=self.dtype,
                            device=self.device,
                        ),
                        torch.zeros(
                            [1, 1, self.ph, self.pw],
                            dtype=self.dtype,
                            device=self.device,
                        ),
                        torch.zeros([2], dtype=torch.float, device=self.device),
                        torch.zeros(
                            [1, 2, self.ph, self.pw],
                            dtype=torch.float,
                            device=self.device,
                        ),
                    ]
                    if self.trt_static_shape:
                        dynamic_shapes = None

                        inputs = [
                            torch_tensorrt.Input(
                                shape=[1, 3, self.ph, self.pw], dtype=self.dtype
                            ),
                            torch_tensorrt.Input(
                                shape=[1, 3, self.ph, self.pw], dtype=self.dtype
                            ),
                            torch_tensorrt.Input(
                                shape=[1, 1, self.ph, self.pw], dtype=self.dtype
                            ),
                            torch_tensorrt.Input(shape=[2], dtype=torch.float),
                            torch_tensorrt.Input(
                                shape=[1, 2, self.ph, self.pw], dtype=torch.float
                            ),
                        ]
                    else:
                        self.trt_min_shape.reverse()
                        self.trt_opt_shape.reverse()
                        self.trt_max_shape.reverse()

                        _height = torch.export.Dim(
                            "height",
                            min=self.trt_min_shape[0] // tmp,
                            max=self.trt_max_shape[0] // tmp,
                        )
                        _width = torch.export.Dim(
                            "width",
                            min=self.trt_min_shape[1] // tmp,
                            max=self.trt_max_shape[1] // tmp,
                        )
                        dim_height = _height * tmp
                        dim_width = _width * tmp
                        dynamic_shapes = {
                            "img0": {2: dim_height, 3: dim_width},
                            "img1": {2: dim_height, 3: dim_width},
                            "timestep": {2: dim_height, 3: dim_width},
                            "tenFlow_div": {},
                            "backwarp_tenGrid": {2: dim_height, 3: dim_width},
                        }

                        inputs = [
                            torch_tensorrt.Input(
                                min_shape=[1, 3] + self.trt_min_shape,
                                opt_shape=[1, 3] + self.trt_opt_shape,
                                max_shape=[1, 3] + self.trt_max_shape,
                                dtype=self.dtype,
                                name="img0",
                            ),
                            torch_tensorrt.Input(
                                min_shape=[1, 3] + self.trt_min_shape,
                                opt_shape=[1, 3] + self.trt_opt_shape,
                                max_shape=[1, 3] + self.trt_max_shape,
                                dtype=self.dtype,
                                name="img1",
                            ),
                            torch_tensorrt.Input(
                                min_shape=[1, 1] + self.trt_min_shape,
                                opt_shape=[1, 1] + self.trt_opt_shape,
                                max_shape=[1, 1] + self.trt_max_shape,
                                dtype=self.dtype,
                                name="timestep",
                            ),
                            torch_tensorrt.Input(
                                shape=[2],
                                dtype=torch.float,
                                name="tenFlow_div",
                            ),
                            torch_tensorrt.Input(
                                min_shape=[1, 2] + self.trt_min_shape,
                                opt_shape=[1, 2] + self.trt_opt_shape,
                                max_shape=[1, 2] + self.trt_max_shape,
                                dtype=torch.float,
                                name="backwarp_tenGrid",
                            ),
                        ]
                    # load encode engine
                    if not self.rife46:
                        exampleInputs += [
                            torch.zeros(
                                (1, num_ch_for_encode, self.ph, self.pw),
                                dtype=self.dtype,
                                device=self.device,
                            ),
                            torch.zeros(
                                (1, num_ch_for_encode, self.ph, self.pw),
                                dtype=self.dtype,
                                device=self.device,
                            ),
                        ]
                        if self.trt_static_shape:
                            inputs += [
                                torch_tensorrt.Input(
                                    shape=[1, 1, self.ph, self.pw], dtype=self.dtype
                                ),
                                torch_tensorrt.Input(
                                    shape=[1, 1, self.ph, self.pw], dtype=self.dtype
                                ),
                            ]
                        else:
                            if not self.trt_static_shape:
                                dynamic_shapes["f0"] = {2: dim_height, 3: dim_width}
                                dynamic_shapes["f1"] = {2: dim_height, 3: dim_width}
                            inputs += [
                                torch_tensorrt.Input(
                                    min_shape=[1, num_ch_for_encode]
                                    + self.trt_min_shape,
                                    opt_shape=[1, num_ch_for_encode]
                                    + self.trt_opt_shape,
                                    max_shape=[1, num_ch_for_encode]
                                    + self.trt_max_shape,
                                    dtype=self.dtype,
                                    name="f0",
                                ),
                                torch_tensorrt.Input(
                                    min_shape=[1, num_ch_for_encode]
                                    + self.trt_min_shape,
                                    opt_shape=[1, num_ch_for_encode]
                                    + self.trt_opt_shape,
                                    max_shape=[1, num_ch_for_encode]
                                    + self.trt_max_shape,
                                    dtype=self.dtype,
                                    name="f1",
                                ),
                            ]

                        encodedExampleInputs = [
                            torch.zeros(
                                (1, 3, self.ph, self.pw),
                                dtype=self.dtype,
                                device=self.device,
                            ),
                        ]
                        encodedInput = [
                            torch_tensorrt.Input(
                                min_shape=[1, 3] + self.trt_min_shape,
                                opt_shape=[1, 3] + self.trt_opt_shape,
                                max_shape=[1, 3] + self.trt_max_shape,
                                dtype=self.dtype,
                                name="x",
                            ),
                        ]

                        if not os.path.isfile(encode_trt_engine_path):
                            printAndLog(
                                "Building TensorRT engine {}".format(trt_engine_path)
                            )
                            if self.trt_static_shape:
                                dynamic_encode_shapes = None
                            else:
                                dynamic_encode_shapes = {
                                    "x": {2: dim_height, 3: dim_width},
                                }
                            exported_encode_program = torch.export.export(
                                self.encode,
                                tuple(encodedExampleInputs),
                                dynamic_shapes=dynamic_encode_shapes,
                            )

                            self.encode = torch_tensorrt.dynamo.compile(
                                exported_encode_program,
                                tuple(encodedInput),
                                device=self.device,
                                enabled_precisions={self.dtype},
                                debug=self.trt_debug,
                                num_avg_timing_iters=4,
                                workspace_size=self.trt_workspace_size,
                                min_block_size=1,
                                max_aux_streams=self.trt_max_aux_streams,
                                optimization_level=self.trt_optimization_level,
                            )
                            printAndLog(
                                f"Saving TensorRT engine to {encode_trt_engine_path}"
                            )
                            torch_tensorrt.save(
                                self.encode,
                                encode_trt_engine_path,
                                output_format="torchscript",
                                inputs=tuple(encodedExampleInputs),
                            )

                        printAndLog(
                            f"Loading TensorRT engine from {encode_trt_engine_path}"
                        )
                        self.encode = torch.jit.load(encode_trt_engine_path).eval()
                        exported_program = torch.export.export(
                            self.flownet,
                            tuple(exampleInputs),
                            dynamic_shapes=dynamic_shapes,
                        )

                    # build flow engine
                    printAndLog("Building TensorRT engine {}".format(trt_engine_path))

                    self.flownet = torch_tensorrt.dynamo.compile(
                        exported_program,
                        tuple(inputs),
                        device=self.device,
                        enabled_precisions={self.dtype},
                        debug=self.trt_debug,
                        num_avg_timing_iters=4,
                        workspace_size=self.trt_workspace_size,
                        min_block_size=1,
                        max_aux_streams=self.trt_max_aux_streams,
                        optimization_level=self.trt_optimization_level,
                    )
                    printAndLog(f"Saving TensorRT engine to {trt_engine_path}")
                    torch_tensorrt.save(
                        self.flownet,
                        trt_engine_path,
                        output_format="torchscript",
                        inputs=tuple(exampleInputs),
                    )
                printAndLog(f"Loading TensorRT engine from {trt_engine_path}")
                self.flownet = torch.jit.load(trt_engine_path).eval()
        self.prepareStream.synchronize()

    @torch.inference_mode()
    def set_rife_args(self):
        self.tenFlow_div = torch.tensor(
            [(self.pw - 1.0) / 2.0, (self.ph - 1.0) / 2.0],
            dtype=torch.float32,
            device=self.device,
        )
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, self.pw, dtype=torch.float32, device=self.device)
            .view(1, 1, 1, self.pw)
            .expand(-1, -1, self.ph, -1)
        ).to(dtype=torch.float32, device=self.device)
        tenVertical = (
            torch.linspace(-1.0, 1.0, self.ph, dtype=torch.float32, device=self.device)
            .view(1, 1, self.ph, 1)
            .expand(-1, -1, -1, self.pw)
        ).to(dtype=torch.float32, device=self.device)
        self.backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1)

    def handlePrecision(self, precision):
        if precision == "auto":
            return torch.float16 if check_bfloat16_support() else torch.float32
        if precision == "float32":
            return torch.float32
        if precision == "float16":
            return torch.float16
        if precision == "bfloat16":
            return torch.bfloat16

    @torch.inference_mode()
    def copyTensor(self, tensorToCopy: torch.Tensor, tensorCopiedTo: torch.Tensor):
        with torch.cuda.stream(self.stream):
            tensorToCopy.copy_(tensorCopiedTo, non_blocking=True)
        self.stream.synchronize()

    def hotUnload(self):
        self.flownet = None
        self.encode = None
        self.tenFlow_div = None
        self.backwarp_tenGrid = None
        self.f0encode = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()

    @torch.inference_mode()
    def hotReload(self):
        self._load()

    @torch.inference_mode()
    def process(self, img0, img1, timestep, f0encode=None, f1encode=None):
        while self.flownet is None:
            sleep(1)
        with torch.cuda.stream(self.stream):
            timestep = torch.full(
                (1, 1, self.ph, self.pw),
                timestep,
                dtype=self.dtype,
                device=self.device,
            ).to(non_blocking=True)
            if not self.rife46:
                output = self.flownet(
                    img0,
                    img1,
                    timestep,
                    self.tenFlow_div,
                    self.backwarp_tenGrid,
                    f0encode,
                    f1encode,
                )
            else:
                output = self.flownet(
                    img0, img1, timestep, self.tenFlow_div, self.backwarp_tenGrid
                )
        self.stream.synchronize()
        return self.tensor_to_frame(output)

    @torch.inference_mode()
    def uncacheFrame(self):
        self.f0encode = None
        self.img0 = None

    @torch.inference_mode()
    def tensor_to_frame(self, frame: torch.Tensor):
        return frame.float().byte().contiguous().cpu().numpy()

    @torch.inference_mode()
    def encode_Frame(self, frame: torch.Tensor):
        while self.encode is None:
            sleep(1)
        with torch.cuda.stream(self.prepareStream):
            frame = self.encode(frame)
        self.prepareStream.synchronize()
        return frame

    @torch.inference_mode()
    def norm(self, frame: torch.Tensor):
        return F.pad(
            frame.reshape(self.height, self.width, 3)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .div(255.0),
            self.padding,
        )

    @torch.inference_mode()
    def frame_to_tensor(self, frame) -> torch.Tensor:
        with torch.cuda.stream(self.prepareStream):
            frame = self.norm(
                torch.frombuffer(
                    frame,
                    dtype=torch.uint8,
                ).to(device=self.device, dtype=self.dtype, non_blocking=True)
            )

        self.prepareStream.synchronize()
        return frame
