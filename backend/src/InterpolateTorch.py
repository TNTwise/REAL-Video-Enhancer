import torch
import torch.nn.functional as F
from .InterpolateArchs.DetectInterpolateArch import ArchDetect
import math
import os
import logging
import gc
from .Util import (
    currentDirectory,
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
try: # temp
    import tensorrt
    import torch_tensorrt
except:
    pass 

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
        self.trt_workspace_size = trt_workspace_size
        self.trt_max_aux_streams = trt_max_aux_streams
        self.trt_optimization_level = trt_optimization_level
        self.trt_cache_dir = trt_cache_dir
        self.backend = backend
        self.ceilInterpolateFactor = ceilInterpolateFactor
        # set up streams for async processing
        self.stream = torch.cuda.Stream()
        self.prepareStream = torch.cuda.Stream()
        self.scale = 1
        self.img0 = None
        self.f0encode = None
        self.rife46 = False
        self.trt_debug = trt_debug
        if UHDMode:
            self.scale = 0.5
        self._load()
    @torch.inference_mode()
    def _load(self):
        with torch.cuda.stream(self.prepareStream):
            state_dict = torch.load(
                self.interpolateModel,
                map_location=self.device,
                weights_only=True,
                mmap=True,
            )

            tmp = max(32, int(32 / self.scale))
            self.pw = math.ceil(self.width / tmp) * tmp
            self.ph = math.ceil(self.height / tmp) * tmp
            self.padding = (0, self.pw - self.width, 0, self.ph - self.height)
            # detect what rife arch to use
            ad = ArchDetect(self.interpolateModel)
            interpolateArch = ad.getArch()
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
            
            self.inputs = [
                torch.zeros((1, 3, self.ph, self.pw), dtype=self.dtype, device=self.device),
                torch.zeros((1, 3, self.ph, self.pw), dtype=self.dtype, device=self.device),
                torch.zeros((1, 1, self.ph, self.pw), dtype=self.dtype, device=self.device),
            ]
            self.encodedInput = [
                torch.zeros((1, 3, self.ph, self.pw), dtype=self.dtype, device=self.device),
            ]
            log("interp arch" + interpolateArch.lower())
            v1=False
            match interpolateArch.lower():
                case "rife46":
                    from .InterpolateArchs.RIFE.rife46IFNET import IFNet
                    self.rife46 = True
                    v1 = True
                case "rife47":
                    from .InterpolateArchs.RIFE.rife47IFNET import IFNet

                    for i in range(2):
                        self.inputs.append(
                            torch.zeros(
                                (1, 4, self.ph, self.pw), dtype=self.dtype, device=self.device
                            ),
                        )
                    self.encode = torch.nn.Sequential(
                        torch.nn.Conv2d(3, 16, 3, 2, 1),
                        torch.nn.ConvTranspose2d(16, 4, 4, 2, 1),
                    )
                case "rife413":
                    from .InterpolateArchs.RIFE.rife413IFNET import IFNet, Head
                    for i in range(2):
                        self.inputs.append(
                            torch.zeros(
                                (1, 8, self.ph, self.pw), dtype=self.dtype, device=self.device
                            ),
                        )
                    v1 = True
                    self.encode = Head()
                case "rife420":
                    from .InterpolateArchs.RIFE.rife420IFNET import IFNet, Head
                    for i in range(2):
                        self.inputs.append(
                            torch.zeros(
                                (1, 8, self.ph, self.pw), dtype=self.dtype, device=self.device
                            ),
                        )
                    
                    self.encode = Head()
                case "rife421":
                    from .InterpolateArchs.RIFE.rife421IFNET import IFNet, Head
                    for i in range(2):
                        self.inputs.append(
                            torch.zeros(
                                (1, 8, self.ph, self.pw), dtype=self.dtype, device=self.device
                            ),
                        )
                    
                    self.encode = Head()
                    v1=True
                case "rife422lite":
                    from .InterpolateArchs.RIFE.rife422_liteIFNET import IFNet, Head
                    for i in range(2):
                        self.inputs.append(
                            torch.zeros(
                                (1, 4, self.ph, self.pw), dtype=self.dtype, device=self.device
                            ),
                        )
                    
                        
                    self.encode = Head()

                    v1 = True
                case _:
                    errorAndLog("Invalid Interpolation Arch")
            self.v1 = v1
            # if 4.6 v1
            if v1:
                self.tenFlow_div = torch.tensor(
                    [(self.pw - 1.0) / 2.0, (self.ph - 1.0) / 2.0],
                    dtype=self.dtype,
                    device=self.device,
                )
                tenHorizontal = (
                    torch.linspace(
                        -1.0, 1.0, self.pw, dtype=self.dtype, device=self.device
                    )
                    .view(1, 1, 1, self.pw)
                    .expand(-1, -1, self.ph, -1)
                ).to(dtype=self.dtype, device=self.device)
                tenVertical = (
                    torch.linspace(
                        -1.0, 1.0, self.ph, dtype=self.dtype, device=self.device
                    )
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
                scale=self.scale,
                ensemble=False,
                dtype=self.dtype,
                device=self.device,
                width=self.width,
                height=self.height,
                backwarp_tenGrid=self.backwarp_tenGrid,
                tenFlow_div=self.tenFlow_div,
            )

            state_dict = {
                k.replace("module.", ""): v
                for k, v in state_dict.items()
                if "module." in k
            }
            head_state_dict =  {
                k.replace("encode.",""): v
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
                torch_tensorrt.runtime.enable_cudagraphs()
                logging.basicConfig(level=logging.INFO)
                trt_engine_path = os.path.join(
                    os.path.realpath(self.trt_cache_dir),
                    (
                        f"{os.path.basename(self.interpolateModel)}"
                        + f"_{self.pw}x{self.ph}"
                        + f"_{'fp16' if self.dtype == torch.float16 else 'fp32'}"
                        + f"_scale-{self.scale}"
                        + f"_ensemble-False"
                        + f"_{torch.cuda.get_device_name(self.device)}"
                        + f"torch_tensorrt-{torch_tensorrt.__version__}"
                        + f"_trt-{tensorrt.__version__}"
                        + (f"rife_version-" + "v1" if v1 else "v2")
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
                        + ".dyn"
                    ),
                )
                if not self.rife46:
                    encode_trt_engine_path = trt_engine_path.replace(".dyn", "_encode.dyn")
                    if not os.path.isfile(encode_trt_engine_path):
                        printAndLog("Building TensorRT engine {}".format(trt_engine_path))

                        self.encode = torch_tensorrt.compile(
                            self.encode,
                            ir="dynamo",
                            inputs=self.encodedInput,
                            enabled_precisions={self.dtype},
                            debug=self.trt_debug,
                            workspace_size=self.trt_workspace_size,
                            min_block_size=1,
                            max_aux_streams=self.trt_max_aux_streams,
                            optimization_level=self.trt_optimization_level,
                            device=self.device,
                            cache_built_engines=False,
                            reuse_cached_engines=False,
                        )
                        printAndLog(f"Saving TensorRT engine to {encode_trt_engine_path}")
                        torch_tensorrt.save(
                            self.encode, encode_trt_engine_path, inputs=self.encodedInput
                        )
                    printAndLog(f"Loading TensorRT engine from {encode_trt_engine_path}")
                    self.encode = torch.export.load(encode_trt_engine_path).module()
                if not os.path.isfile(trt_engine_path):
                    printAndLog("Building TensorRT engine {}".format(trt_engine_path))

                    self.flownet = torch_tensorrt.compile(
                        self.flownet,
                        ir="dynamo",
                        inputs=self.inputs,
                        enabled_precisions={self.dtype},
                        debug=self.trt_debug,
                        workspace_size=self.trt_workspace_size,
                        min_block_size=1,
                        max_aux_streams=self.trt_max_aux_streams,
                        optimization_level=self.trt_optimization_level,
                        device=self.device,
                        cache_built_engines=False,
                        reuse_cached_engines=False,
                    )
                    printAndLog(f"Saving TensorRT engine to {trt_engine_path}")
                    torch_tensorrt.save(
                        self.flownet, trt_engine_path, inputs=self.inputs
                    )
                printAndLog(f"Loading TensorRT engine from {trt_engine_path}")
                self.flownet = torch.export.load(trt_engine_path).module()

    def handlePrecision(self, precision):
        if precision == "auto":
            return torch.float16 if check_bfloat16_support() else torch.float32
        if precision == "float32":
            return torch.float32
        if precision == "float16":
            return torch.float16

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
    def process(self, img0, img1, timestep):
        while self.flownet is None:
            sleep(1)
        with torch.cuda.stream(self.stream):
                timestep = self.timestepDict[timestep]
                if self.img0 is None:
                    self.img0 = self.frame_to_tensor(img0)
                img1 = self.frame_to_tensor(img1)
                if not self.rife46:
                    if self.f0encode is None:
                        self.f0encode = self.encode(self.img0[:, :3])
                    f1encode = self.encode(img1[:, :3])
                    output = self.flownet(
                        self.img0, img1, timestep, self.f0encode, f1encode
                    )
                    self.f0encode.copy_(f1encode,non_blocking=True)
                else:
                    output = self.flownet(img0, img1, timestep)
                self.img0.copy_(img1,non_blocking=True)
        self.stream.synchronize()
        return self.tensor_to_frame(output)

    @torch.inference_mode()
    def uncacheFrame(self, n):
        self.f0encode = None
        self.img0 = None

    @torch.inference_mode()
    def tensor_to_frame(self, frame: torch.Tensor):
        """
        Takes in a 4d tensor, undoes padding, and converts to np array for rendering
        """

        return frame.byte().contiguous().cpu().numpy()
    
    @torch.inference_mode()
    def encode_Frame(self,frame:torch.Tensor):
        return self.encode(frame[:, :3])

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
