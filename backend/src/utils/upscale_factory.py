class UpscaleFactory:
    """
    Factory class to create the correct upscale implementation based on the selected backend.

    Backends:
        - pytorch / tensorrt: uses the PyTorch (or TensorRT) upscale implementation.
        - ncnn: uses the NCNN upscale implementation.
        - directml: uses the ONNX / DirectML upscale implementation.

    Adding support for a new backend/model family only requires adding a new branch here,
    the Render class itself does not need to know about backend specifics.
    """

    @staticmethod
    def build_upscale(
        backend: str,
        modelPath: str,
        device: str = "default",
        precision: str = "float16",
        width: int = 1920,
        height: int = 1080,
        tilesize: int = 0,
        gpu_id: int = 0,
        hdr_mode: bool = False,
        trt_optimization_level: int = 3,
        trt_dynamic_shapes: bool = False,
        scale: int = None,
        num_threads: int = 1,
        tile_pad: int = 10,
    ):
        """
        Builds the upscale implementation used to upscale a frame for the given backend.

        Args:
            backend (str): backend used for inference (pytorch/ncnn/tensorrt/directml).
            modelPath (str): path to the upscale model.
            device (str, optional): device used for inference. Defaults to "default".
            precision (str, optional): precision used for inference. Defaults to "float16".
            width (int, optional): width of the frame being upscaled. Defaults to 1920.
            height (int, optional): height of the frame being upscaled. Defaults to 1080.
            tilesize (int, optional): tile size used for tiled upscaling, 0 for single shot. Defaults to 0.
            gpu_id (int, optional): gpu id used for inference. Defaults to 0.
            hdr_mode (bool, optional): whether the frames are hdr. Defaults to False.
            trt_optimization_level (int, optional): tensorrt optimization level. Defaults to 3.
            trt_dynamic_shapes (bool, optional): whether tensorrt uses dynamic shapes. Defaults to False.
            scale (int, optional): upscale scale to use, if None the model scale is used. Defaults to None.
            num_threads (int, optional): number of threads used by ncnn. Defaults to 1.
            tile_pad (int, optional): tile padding used by ncnn. Defaults to 10.

        Returns:
            The constructed upscale inference object, with getScale, hotUnload, hotReload and __call__.
        """
        if backend in ("pytorch", "tensorrt"):

            from ..pytorch.upscale_torch import UpscalePytorch

            return UpscalePytorch(
                modelPath,
                device=device,
                precision=precision,
                width=width,
                height=height,
                backend=backend,
                tilesize=tilesize,
                gpu_id=gpu_id,
                hdr_mode=hdr_mode,
                trt_optimization_level=trt_optimization_level,
                trt_static_shape=not trt_dynamic_shapes,
            )

        if backend == "ncnn":

            from ..ncnn.upscale_ncnn import UpscaleNCNN, getNCNNScale

            return UpscaleNCNN(
                modelPath=modelPath,
                num_threads=num_threads,
                scale=scale if scale is not None else getNCNNScale(modelPath=modelPath),
                gpuid=gpu_id,  # might have this be a setting
                width=width,
                height=height,
                tilesize=tilesize,
                tilePad=tile_pad,
                hdr_mode=hdr_mode,
            )

        if backend == "directml":

            from ..onnx.upscale_onnx import UpscaleONNX

            return UpscaleONNX(
                modelPath=modelPath,
                deviceID=gpu_id,
                precision=precision,
                width=width,
                height=height,
                scale=scale if scale is not None else UpscaleONNX.getModelScale(modelPath),
                gpu_id=gpu_id,
                hdr_mode=hdr_mode,
            )

        raise ValueError(f"Unsupported backend for upscaling model: {backend}")
