import torch
import torch.nn.functional as F
import math
import os

from .InterpolateArchs.DetectInterpolateArch import loadInterpolationModel
from .Util import currentDirectory

torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)

class InterpolateRifeTorch:
    @torch.inference_mode()
    def __init__(
        self,
        interpolateModelPath: str,
        width: int = 1920,
        height: int = 1080,
        device: str = "cuda",
        dtype: str = "float16",
        backend: str = "pytorch",
        UHDMode: bool = False,
        ensemble: bool = False,
        # trt options
        trt_min_shape: list[int] = [128, 128],
        trt_opt_shape: list[int] = [1920, 1080],
        trt_max_shape: list[int] = [1920, 1080],
        trt_workspace_size: int = 0,
        trt_max_aux_streams: int | None = None,
        trt_optimization_level: int = 5,
        trt_cache_dir: str = currentDirectory(),
        trt_debug: bool = False,
    ):
        trt_min_shape = [int(width/15),int(height/15)]
        trt_opt_shape = [width,height]
        trt_max_shape = [width,height]

        self.interpolateModel = interpolateModelPath
        self.width = width
        self.height = height
        self.device = torch.device(device, 0) # 0 is the device index, may have to change later
        self.dtype = self.handlePrecision(dtype)
        self.backend = backend
        scale = 1
        if UHDMode:
            scale = 0.5

        state_dict = torch.load(
            interpolateModelPath, map_location=self.device, weights_only=True, mmap=True
        )

        # detect what rife arch to use
        model = loadInterpolationModel(state_dict)
        architecture = model.getIFnet()
        self.flownet = architecture(scale=scale, ensemble=ensemble)

        state_dict = {
            k.replace("module.", ""): v for k, v in state_dict.items() if "module." in k
        }
        self.flownet.load_state_dict(
            state_dict=state_dict, strict=False
        )
        self.flownet.eval().to(device=self.device)
        if self.dtype == torch.float16:
            self.flownet.half()

        tmp = max(32, int(32 / scale))
        self.pw = math.ceil(self.width / tmp) * tmp
        self.ph = math.ceil(self.height / tmp) * tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)
        self.tenFlow_div = torch.tensor(
            [(self.pw - 1.0) / 2.0, (self.ph - 1.0) / 2.0],
            dtype=self.dtype,
            device=self.device,
        )

        tenHorizontal = (
            torch.linspace(-1.0, 1.0, self.pw, dtype=self.dtype, device=self.device)
            .view(1, 1, 1, self.pw)
            .expand(-1, -1, self.ph, -1)
        )
        tenVertical = (
            torch.linspace(-1.0, 1.0, self.ph, dtype=self.dtype, device=self.device)
            .view(1, 1, self.ph, 1)
            .expand(-1, -1, -1, self.pw)
        )
        self.backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1)

        if self.backend == "tensorrt":
            import tensorrt
            import torch_tensorrt

            for i in range(2):
                trt_min_shape[i] = math.ceil(max(trt_min_shape[i], 1) / tmp) * tmp
                trt_opt_shape[i] = math.ceil(max(trt_opt_shape[i], 1) / tmp) * tmp
                trt_max_shape[i] = math.ceil(max(trt_max_shape[i], 1) / tmp) * tmp

            dimensions = (
                f"min-{trt_min_shape[0]}x{trt_min_shape[1]}"
                f"_opt-{trt_opt_shape[0]}x{trt_opt_shape[1]}"
                f"_max-{trt_max_shape[0]}x{trt_max_shape[1]}"
            )
            trt_engine_path = os.path.join(
                os.path.realpath(trt_cache_dir),
                (
                    f"{os.path.basename(self.interpolateModel)}"
                    + f"_{dimensions}"
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
                trt_min_shape.reverse()
                trt_opt_shape.reverse()
                trt_max_shape.reverse()

                example_tensors = (
                    torch.zeros((1, 3, self.ph, self.ph), dtype=self.dtype, device=self.device),
                    torch.zeros((1, 3, self.ph, self.ph), dtype=self.dtype, device=self.device),
                    torch.zeros((1, 1, self.ph, self.ph), dtype=self.dtype, device=self.device),
                    torch.zeros((2,), dtype=self.dtype, device=self.device),
                    torch.zeros((1, 2, self.ph, self.ph), dtype=self.dtype, device=self.device),
                )

                _height = torch.export.Dim(
                    "height", min=trt_min_shape[0] // tmp, max=trt_max_shape[0] // tmp
                )
                _width = torch.export.Dim(
                    "width", min=trt_min_shape[1] // tmp, max=trt_max_shape[1] // tmp
                )
                dim_height = _height * tmp
                dim_width = _width * tmp
                dynamic_shapes = {
                    "img0": {2: dim_height, 3: dim_width},
                    "img1": {2: dim_height, 3: dim_width},
                    "timestep": {2: dim_height, 3: dim_width},
                    "tenFlow_div": {0: None},
                    "backwarp_tenGrid": {2: dim_height, 3: dim_width},
                }

                exported_program = torch.export.export(
                    self.flownet, example_tensors, dynamic_shapes=dynamic_shapes
                )

                inputs = [
                    torch_tensorrt.Input(
                        min_shape=[1, 3] + trt_min_shape,
                        opt_shape=[1, 3] + trt_opt_shape,
                        max_shape=[1, 3] + trt_max_shape,
                        dtype=self.dtype,
                        name="img0",
                    ),
                    torch_tensorrt.Input(
                        min_shape=[1, 3] + trt_min_shape,
                        opt_shape=[1, 3] + trt_opt_shape,
                        max_shape=[1, 3] + trt_max_shape,
                        dtype=self.dtype,
                        name="img1",
                    ),
                    torch_tensorrt.Input(
                        min_shape=[1, 1] + trt_min_shape,
                        opt_shape=[1, 1] + trt_opt_shape,
                        max_shape=[1, 1] + trt_max_shape,
                        dtype=self.dtype,
                        name="timestep",
                    ),
                    torch_tensorrt.Input(
                        shape=[2],
                        dtype=self.dtype,
                        name="tenFlow_div",
                    ),
                    torch_tensorrt.Input(
                        min_shape=[1, 2] + trt_min_shape,
                        opt_shape=[1, 2] + trt_opt_shape,
                        max_shape=[1, 2] + trt_max_shape,
                        dtype=self.dtype,
                        name="backwarp_tenGrid",
                    ),
                ]

                flownet = torch_tensorrt.dynamo.compile(
                    exported_program,
                    inputs,
                    enabled_precisions={self.dtype},
                    debug=trt_debug,
                    workspace_size=trt_workspace_size,
                    min_block_size=1,
                    max_aux_streams=trt_max_aux_streams,
                    optimization_level=trt_optimization_level,
                    device=self.device,
                    assume_dynamic_shape_support=True,
                )

                torch_tensorrt.save(
                    flownet,
                    trt_engine_path,
                    output_format="torchscript",
                    inputs=example_tensors,
                )

            self.flownet = torch.jit.load(trt_engine_path).eval()

    def handlePrecision(self, precision):
        if precision == "float32":
            return torch.float32
        if precision == "float16":
            return torch.float16

    @torch.inference_mode()
    def process(self, img0, img1, timestep):
        if timestep == 1:
            return self.tensor_to_frame(img1[:, :, : self.height, : self.width][0])
        if timestep == 0:
            return self.tensor_to_frame(img0[:, :, : self.height, : self.width][0])
        

        timestep = torch.full(
            (1, 1, self.ph, self.pw), timestep, dtype=self.dtype, device=self.device
        )


        output = self.flownet(
            img0, img1, timestep, self.tenFlow_div, self.backwarp_tenGrid
        )
        output = output[:, :, : self.height, : self.width]
        return self.tensor_to_frame(output[0])

    @torch.inference_mode()
    def tensor_to_frame(self, frame: torch.Tensor):
        return (
            frame.squeeze(0)
            .permute(1, 2, 0)
            .float()
            .mul(255)
            .byte()
            .contiguous()
            .cpu()
            .numpy()
        )

    @torch.inference_mode()
    def frame_to_tensor(self, frame) -> torch.Tensor:
        frame = torch.frombuffer(frame, dtype=torch.uint8).reshape(
            self.height, self.width, 3
        )
        return F.pad((frame).permute(2, 0, 1).unsqueeze(0).to(
            self.device, dtype=self.dtype
        ) / 255.0, self.padding)
        