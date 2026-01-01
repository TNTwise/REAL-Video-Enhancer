import torch
import torch.nn.functional as F
from typing import Generator

# from backend.src.pytorch.InterpolateArchs.GIMM import GIMM
from .BaseInterpolate import BaseInterpolate, DynamicScale
from .InterpolateArchs.DetectInterpolateArch import ArchDetect
from .DRBA.infer import DRBA_RVE
from .TorchUtils import TorchUtils
import math
import os
import sys
from ..utils.Util import (
    errorAndLog, log
)
from ..utils.Frame import Frame
from time import sleep

torch.set_float32_matmul_precision("medium")
torch.set_grad_enabled(False)

class InterpolateRifeTorch(BaseInterpolate):
    @torch.inference_mode()
    def __init__(
        self,
        modelPath: str,
        ceilInterpolateFactor: int = 2,
        width: int = 1920,
        height: int = 1080,
        device: str = "auto",
        dtype: str = "auto",
        backend: str = "pytorch",
        UHDMode: bool = False,
        ensemble: bool = False,
        dynamicScaledOpticalFlow: bool = False,
        drba:bool = False,
        gpu_id: int = 0,
        # trt options
        trt_optimization_level: int = 5,
        hdr_mode: bool = False,
        trt_static_shape: bool = False,
        
        *args,
        **kwargs,
    ):

        self.interpolateModel = modelPath
        self.width = width
        self.height = height
        self.device_type = device
        self.device: torch.device = TorchUtils.handle_device(device, gpu_id=gpu_id)
        self.dtype = TorchUtils.handle_precision(dtype)
        self.backend = backend
        self.ceilInterpolateFactor = ceilInterpolateFactor
        self.dynamicScaledOpticalFlow = dynamicScaledOpticalFlow
        
        if width <= 3840 and height <= 3840 and (width > 1920 or height > 1920):
            self.trt_min_shape = ([1920, height] if height < width else [width, 1920]) if width < 1920 or height < 1920 else [1920, 1920]
            self.trt_opt_shape = [3840, 2160]
            self.trt_max_shape = [3840, 3840]
        
        if width <= 1920 and height <= 1920 and (width >= 128 or height >= 128):
            
            self.trt_min_shape = [128, 128]
            self.trt_opt_shape = [1920, 1080]
            self.trt_max_shape = [1920, 1920]
        
        if width > 3840 or height > 3840 and not trt_static_shape:
            log("The video resolution is very large for TensorRT dynamic shape, falling back to static shape")
            trt_static_shape = True

        if width < 128 or height < 128 and not trt_static_shape:
            log("The video resolution is too small for TensorRT dynamic shape, falling back to static shape")
            trt_static_shape = True

        self.trt_static_shape = trt_static_shape
        
        self.CompareNet = None
        self.frame0 = None
        self.encode0 = None
        # set up streams for async processing
        self.scale = 1
        self.ensemble = ensemble
        self.hdr_mode = hdr_mode # used in base interpolate class (ik inheritance is bad leave me alone)

        self.trt_optimization_level = trt_optimization_level
        self.trt_cache_dir = os.path.dirname(
            modelPath
        )  # use the model directory as the cache directory
        self.UHDMode = UHDMode
        if self.UHDMode:
            print("UHD Mode has been depricated for RIFE.", file=sys.stderr) # causes issues with 4k warp.
            self.scale = 1
        
        if drba:
            fps = 24
            self.drba = DRBA_RVE(model_type="rife", model_path="./flownet.pkl", times=ceilInterpolateFactor, dst_fps=fps*ceilInterpolateFactor, fps=fps, scale=1)
        self._load()

    @torch.inference_mode()
    def _load(self):
            

        state_dict = torch.load(
            self.interpolateModel,
            map_location=self.device,
            weights_only=True,
            mmap=True,
        )
        # detect what rife arch to use

        ad = ArchDetect(self.interpolateModel)
        interpolateArch = ad.getArchName()
        _pad = 32
        num_ch_for_encode = 0
        self.encode = None

        match interpolateArch.lower():
            case "rife46":
                from .InterpolateArchs.RIFE.rife46IFNET import IFNet
            case "rife47":
                from .InterpolateArchs.RIFE.rife47IFNET import IFNet

                num_ch_for_encode = 4
                self.encode = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 16, 3, 2, 1),
                    torch.nn.ConvTranspose2d(16, 4, 4, 2, 1),
                ).float()
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
            case "rife425_heavy":
                from .InterpolateArchs.RIFE.rife425_heavyIFNET import IFNet, Head
                _pad = 64
                num_ch_for_encode = 16
                self.encode = Head()

            case _:
                errorAndLog("Invalid Interpolation Arch")
                exit()

        # model unspecific setup
        if self.dynamicScaledOpticalFlow:
            tmp = max(
                _pad, int(_pad / 0.25)
            )  # set pad to higher for better dynamic optical scale support
        else:
            tmp = max(_pad, int(_pad / self.scale))

        self.pw = math.ceil(self.width / tmp) * tmp
        self.ph = math.ceil(self.height / tmp) * tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)
        need_pad = any(p > 0 for p in self.padding)
        self.torchUtils = TorchUtils(
                        width=self.width,
                        height=self.height,
                        hdr_mode=self.hdr_mode,
                        device_type=self.device_type,
                        )
        
        self.stream = self.torchUtils.init_stream()
        self.prepareStream = self.torchUtils.init_stream()
        self.copyStream = self.torchUtils.init_stream()
        self.f2tStream = self.torchUtils.init_stream()

        # caching the timestep tensor in a dict with the timestep as a float for the key

        self.timestepDict = {}
        for n in range(self.ceilInterpolateFactor):
            timestep = n / (self.ceilInterpolateFactor)
            timestep_tens = torch.full(
                (1, 1, self.ph, self.pw),
                timestep,
                dtype=self.dtype,
                device=self.device,
            )
            self.timestepDict[timestep] = timestep_tens
        # rife specific setup

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


        self.flownet = IFNet(
            scale=self.scale,
            ensemble=self.ensemble,
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
        if self.encode:
            self.encode.load_state_dict(state_dict=head_state_dict, strict=True)
            self.encode.eval().to(device=self.device, dtype=self.dtype)
        self.flownet.load_state_dict(state_dict=state_dict, strict=False)
        self.flownet.eval().to(device=self.device, dtype=self.dtype)

        if self.dynamicScaledOpticalFlow:
            if self.backend == "tensorrt":
                print(
                    "Dynamic Scaled Optical Flow does not work with TensorRT, disabling",
                    file=sys.stderr,
                )

            elif self.UHDMode:
                print(
                    "Dynamic Scaled Optical Flow does not work with UHD Mode, disabling",
                    file=sys.stderr,
                )
            else:
                from ..utils.SSIM import SSIM

                CompareNet = SSIM().to(device=self.device, dtype=self.dtype)
                possible_values = {
                    0.25: 0.25,
                    0.37: 0.5,
                    0.5: 1.0,
                    0.69: 1.5,
                    1.0: 2.0,
                }  # closest_value:representative_scale
                self.dynamicScale = DynamicScale(
                    possible_values=possible_values, CompareNet=CompareNet
                )
                print("Dynamic Scaled Optical Flow Enabled")

        if self.backend == "tensorrt":
            import tensorrt # import just in case of error
            import torch_tensorrt
            from .TensorRTHandler import TorchTensorRTHandler

            trtHandler = TorchTensorRTHandler(
                model_parent_path=os.path.dirname(self.interpolateModel),
                trt_optimization_level=self.trt_optimization_level,
            )

            if self.trt_static_shape:
                dimensions = f"{self.width}x{self.height}"
            else:
                for i in range(2):
                    self.trt_min_shape[i] = math.ceil(self.trt_min_shape[i] / tmp) * tmp
                    self.trt_opt_shape[i] = math.ceil(self.trt_opt_shape[i] / tmp) * tmp
                    self.trt_max_shape[i] = math.ceil(self.trt_max_shape[i] / tmp) * tmp

                dimensions = (
                    f"min-{self.trt_min_shape[0]}x{self.trt_min_shape[1]}"
                    f"_opt-{self.trt_opt_shape[0]}x{self.trt_opt_shape[1]}"
                    f"_max-{self.trt_max_shape[0]}x{self.trt_max_shape[1]}"
                )
            base_trt_engine_name = os.path.join(
                (
                    f"{os.path.basename(self.interpolateModel)}"
                    + f"_{dimensions}"
                    + f"_{'fp16' if self.dtype == torch.float16 else 'fp32'}"
                    + f"_scale-{self.scale}"
                    + f"_{torch.cuda.get_device_name(self.device)}"
                    + f"_trt-{trtHandler.tensorrt_version}"
                    + f"_ensemble-{self.ensemble}"
                    + f"_torch_tensorrt-{trtHandler.torch_tensorrt_version}"
                    + (
                        f"_level-{self.trt_optimization_level}"
                        if self.trt_optimization_level is not None
                        else ""
                    )
                ),
            )
            encode_trt_engine_name = base_trt_engine_name + "_encode"

            # lay out inputs
            # load flow engine
            if not trtHandler.check_engine_exists(base_trt_engine_name):
                if self.trt_static_shape:
                    if self.encode:
                        flownet_inputs = (
                            torch.zeros([1, 3, self.ph, self.pw], dtype=self.dtype, device=self.device),
                            torch.zeros([1, 3, self.ph, self.pw], dtype=self.dtype, device=self.device),
                            torch.zeros([1, 1, self.ph, self.pw], dtype=self.dtype, device=self.device),
                            torch.zeros([2], dtype=torch.float, device=self.device),
                            torch.zeros([1, 2, self.ph, self.pw], dtype=torch.float, device=self.device),
                            torch.zeros([1, num_ch_for_encode, self.ph, self.pw], dtype=self.dtype, device=self.device),
                            torch.zeros([1, num_ch_for_encode, self.ph, self.pw], dtype=self.dtype, device=self.device),
                        )

                        encode_inputs = (torch.zeros([1, 3, self.ph, self.pw], dtype=self.dtype, device=self.device),)

                    else:
                        flownet_inputs = (
                            torch.zeros([1, 3, self.ph, self.pw], dtype=self.dtype, device=self.device),
                            torch.zeros([1, 3, self.ph, self.pw], dtype=self.dtype, device=self.device),
                            torch.zeros([1, 1, self.ph, self.pw], dtype=self.dtype, device=self.device),
                            torch.zeros([2], dtype=torch.float, device=self.device),
                            torch.zeros([1, 2, self.ph, self.pw], dtype=torch.float, device=self.device),
                        )
                    flownet_dynamic_shapes = None
                    encode_dynamic_shapes = None

                else:
                    self.trt_min_shape.reverse()
                    self.trt_opt_shape.reverse()
                    self.trt_max_shape.reverse()
                    if self.encode is not None:
                        flownet_inputs = (
                            torch.zeros([1, 3] + self.trt_opt_shape, dtype=self.dtype, device=self.device),
                            torch.zeros([1, 3] + self.trt_opt_shape, dtype=self.dtype, device=self.device),
                            torch.zeros([1, 1] + self.trt_opt_shape, dtype=self.dtype, device=self.device),
                            torch.zeros([2], dtype=torch.float, device=self.device),
                            torch.zeros([1, 2] + self.trt_opt_shape, dtype=torch.float, device=self.device),
                            torch.zeros([1, num_ch_for_encode] + self.trt_opt_shape, dtype=self.dtype, device=self.device),
                            torch.zeros([1, num_ch_for_encode] + self.trt_opt_shape, dtype=self.dtype, device=self.device),
                        )

                        encode_inputs = (torch.zeros([1, 3] + self.trt_opt_shape, dtype=self.dtype, device=self.device),)
                    else:
                        flownet_inputs = (
                            torch.zeros([1, 3] + self.trt_opt_shape, dtype=self.dtype, device=self.device),
                            torch.zeros([1, 3] + self.trt_opt_shape, dtype=self.dtype, device=self.device),
                            torch.zeros([1, 1] + self.trt_opt_shape, dtype=self.dtype, device=self.device),
                            torch.zeros([2], dtype=torch.float, device=self.device),
                            torch.zeros([1, 2] + self.trt_opt_shape, dtype=torch.float, device=self.device),
                        )

                    _height = torch.export.Dim("height", min=self.trt_min_shape[0] // tmp, max=self.trt_max_shape[0] // tmp)
                    _width = torch.export.Dim("width", min=self.trt_min_shape[1] // tmp, max=self.trt_max_shape[1] // tmp)
                    dim_height = _height * tmp
                    dim_width = _width * tmp
                    if self.encode is not None:
                        flownet_dynamic_shapes = {
                            "img0": {2: dim_height, 3: dim_width},
                            "img1": {2: dim_height, 3: dim_width},
                            "timestep": {2: dim_height, 3: dim_width},
                            "tenFlow_div": {},
                            "backwarp_tenGrid": {2: dim_height, 3: dim_width},
                            "f0": {2: dim_height, 3: dim_width},
                            "f1": {2: dim_height, 3: dim_width},
                        }

                        encode_dynamic_shapes = ({2: dim_height, 3: dim_width},)
                    else:
                        flownet_dynamic_shapes = {
                            "img0": {2: dim_height, 3: dim_width},
                            "img1": {2: dim_height, 3: dim_width},
                            "timestep": {2: dim_height, 3: dim_width},
                            "tenFlow_div": {},
                            "backwarp_tenGrid": {2: dim_height, 3: dim_width},
                        }
                
                flownet_engine = trtHandler.build_engine(self.flownet, self.dtype, self.device, flownet_inputs, trt_engine_name=base_trt_engine_name, trt_multi_precision_engine=True, dynamic_shapes=flownet_dynamic_shapes,)
                trtHandler.save_engine(flownet_engine, base_trt_engine_name, flownet_inputs)
                TorchUtils.clear_cache()
                if self.encode:
                    encode_engine = trtHandler.build_engine(self.encode, self.dtype, self.device, encode_inputs, trt_engine_name=encode_trt_engine_name, trt_multi_precision_engine=True, dynamic_shapes=encode_dynamic_shapes,)
                    trtHandler.save_engine(encode_engine, encode_trt_engine_name, encode_inputs)
                TorchUtils.clear_cache()
            self.flownet = trtHandler.load_engine(base_trt_engine_name)

            if self.encode:
                self.encode = trtHandler.load_engine(encode_trt_engine_name)

        
        self.torchUtils.sync_all_streams()

    def debug_save_tensor_as_img(self, img: torch.Tensor, name: str):
        import cv2
        img = img.squeeze().permute(1,2,0).detach().cpu().numpy() * 255
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(name, img)


    @torch.inference_mode()
    def __call__(
        self,
        img1: Frame,
        transition=False,
    ) -> Generator[Frame, Frame, Frame]:  
        
        with self.torchUtils.run_stream(self.stream):  # type: ignore
            with self.torchUtils.run_stream(self.prepareStream):  # type: ignore
                if self.frame0 is None:
                    self.frame0 = F.pad(img1.get_frame_tensor(),self.padding)
                    if self.encode:
                        self.encode0 = self.encode_Frame(self.frame0, self.prepareStream)
                    return
            
                frame1 = F.pad(img1.get_frame_tensor(),self.padding)
            self.torchUtils.sync_stream(self.prepareStream)
            
            if self.encode:
                encode1 = self.encode_Frame(frame1, self.f2tStream)


            if self.dynamicScaledOpticalFlow:
                closest_value = self.dynamicScale.dynamicScaleCalculation(
                    self.frame0, frame1
                )
            else:
                closest_value = None

            for n in range(self.ceilInterpolateFactor - 1):
                if not transition:
                    timestep = (n + 1) * 1.0 / (self.ceilInterpolateFactor)
                    while self.flownet is None:
                        sleep(1)
                    timestep = self.timestepDict[timestep]
                    if self.backend == "pytorch":
                        if self.encode:
                            output = self.flownet(
                                self.frame0,
                                frame1,
                                timestep,
                                self.tenFlow_div,
                                self.backwarp_tenGrid,
                                self.encode0,
                                encode1,  # type: ignore
                                closest_value,
                            )
                        else:
                            output = self.flownet(
                                self.frame0,
                                frame1,
                                timestep,
                                self.tenFlow_div,
                                self.backwarp_tenGrid,
                                closest_value,
                            )
                        # self.debug_save_tensor_as_img(output, f"output/output{n}.png")
                    else:
                        if self.encode:
                            output = self.flownet(
                                self.frame0,
                                frame1,
                                timestep,
                                self.tenFlow_div,
                                self.backwarp_tenGrid,
                                self.encode0,
                                encode1,  # type: ignore
                            )
                        else:
                            output = self.flownet(
                                self.frame0,
                                frame1,
                                timestep,
                                self.tenFlow_div,
                                self.backwarp_tenGrid,
                            )
                    
                    yield img1.get_dummy_frame().set_frame_tensor(output[:, :, : self.height, : self.width])

                else:
                    yield img1


            self.torchUtils.copy_tensor(self.frame0, frame1, self.copyStream)
            if self.encode:
                self.torchUtils.copy_tensor(self.encode0, encode1, self.copyStream)  # type: ignore

            # self.debug_save_tensor_as_img(self.frame0, "frame0.png")


        self.torchUtils.sync_all_streams()

    @torch.inference_mode()
    def encode_Frame(self, frame: torch.Tensor, stream: torch.Stream):
        while self.encode is None:
            sleep(1)
        with self.torchUtils.run_stream(stream):  # type: ignore
            frame = self.encode(frame)
        self.torchUtils.sync_stream(stream)
        return frame



class InterpolateRIFEDRBA(InterpolateRifeTorch):
    
    @torch.inference_mode()
    def __call__(
        self,
        img1,
        transition=False,
    ):
        if self.frame0 is None:
            self.frame0 = img1
            out = self.drba.header(self.frame0, img1)
            yield self.tensor_to_frame(out)
        with torch.cuda.stream(self.stream):  # type: ignore
            out = self.drba.inference(img1)
            yield self.tensor_to_frame(out)