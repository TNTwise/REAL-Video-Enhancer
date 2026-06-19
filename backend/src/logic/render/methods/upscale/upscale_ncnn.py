class UPSCALE:
    def __init__(
        self,
        gpuid: int = 0,
        tta_mode: bool = False,
        tilesize: int = 0,
        model: int = 0,
        num_threads: int = 1,
        model_str: str = "",
        scale: int = 0,
    ):
        assert gpuid >= -1, "gpuid must >= -1"
        assert tilesize == 0 or tilesize >= 32, "tilesize must >= 32 or be 0"
        assert model >= -1, "model must > 0 or -1"
        assert num_threads >= 1, "num_threads must be a positive integer"
        self._gpuid = gpuid
        self._model_str = model_str
        self._upscale_object = wrapped.UPSCALEWrapped(gpuid, tta_mode, num_threads)

        self._tilesize = tilesize
        self._model = model
        self._scale = scale

        if self._model > -1:
            self._load(scale=scale)

        self.raw_in_image = None
        self.raw_out_image = None

        self.channels = None
        self.out_bytes = None

    def _set_parameters(self) -> None:
        self._upscale_object.set_parameters(self._tilesize, self._scale)

    def _load(
        self,
        param_path: Optional[pathlib.Path] = None,
        model_path: Optional[pathlib.Path] = None,
        scale: int = 0,
    ) -> None:
        model_dict: Dict[int, Dict[str, Union[str, int]]] = {
            # span
            0: {
                "param": "spanx2_ch48.param",
                "bin": "spanx2_ch48.bin",
                "scale": 2,
                "folder": "models/SPAN",
            },
            1: {
                "param": "spanx2_ch52.param",
                "bin": "spanx2_ch52.bin",
                "scale": 2,
                "folder": "models/SPAN",
            },
            2: {
                "param": "spanx4_ch48.param",
                "bin": "spanx4_ch48.bin",
                "scale": 4,
                "folder": "models/SPAN",
            },
            3: {
                "param": "spanx4_ch52.param",
                "bin": "spanx4_ch52.bin",
                "scale": 4,
                "folder": "models/SPAN",
            },
            # custom span
            4: {
                "param": "2x_ModernSpanimationV1.param",
                "bin": "2x_ModernSpanimationV1.bin",
                "scale": 2,
                "folder": "models/SPAN",
            },
            5: {
                "param": "4xSPANkendata.param",
                "bin": "4xSPANkendata.bin",
                "scale": 4,
                "folder": "models/SPAN",
            },
            6: {
                "param": "ClearReality4x.param",
                "bin": "ClearReality4x.bin",
                "scale": 4,
                "folder": "models/SPAN",
            },
            # esrgan
            7: {
                "param": "realesr-animevideov3-x2.param",
                "bin": "realesr-animevideov3-x2.bin",
                "scale": 2,
                "folder": "models/ESRGAN",
            },
            8: {
                "param": "realesr-animevideov3-x3.param",
                "bin": "realesr-animevideov3-x3.bin",
                "scale": 3,
                "folder": "models/ESRGAN",
            },
            9: {
                "param": "realesr-animevideov3-x4.param",
                "bin": "realesr-animevideov3-x4.bin",
                "scale": 4,
                "folder": "models/ESRGAN",
            },
            10: {
                "param": "realesrgan-x4plus-x4.param",
                "bin": "realesrgan-x4plus.bin",
                "scale": 4,
                "folder": "models/ESRGAN",
            },
            11: {
                "param": "realesrgan-x4plus-anime.param",
                "bin": "realesrgan-x4plus-anime.bin",
                "scale": 4,
                "folder": "models/ESRGAN",
            },
            # cugan-se models
            12: {
                "param": "up2x-conservative.param",
                "bin": "up2x-conservative.bin",
                "scale": 2,
                "folder": "models/CUGAN/models-se",
            },
            13: {
                "param": "up2x-no-denoise.param",
                "bin": "up2x-no-denoise.bin",
                "scale": 2,
                "folder": "models/CUGAN/models-se",
            },
            14: {
                "param": "up2x-denoise1x.param",
                "bin": "up2x-denoise1x.bin",
                "scale": 2,
                "folder": "models/CUGAN/models-se",
            },
            15: {
                "param": "up2x-denoise2x.param",
                "bin": "up2x-denoise2x.bin",
                "scale": 2,
                "folder": "models/CUGAN/models-se",
            },
            16: {
                "param": "up2x-denoise3x.param",
                "bin": "up2x-denoise3x.bin",
                "scale": 2,
                "folder": "models/CUGAN/models-se",
            },
            17: {
                "param": "up3x-conservative.param",
                "bin": "up3x-conservative.bin",
                "scale": 3,
                "folder": "models/CUGAN/models-se",
            },
            18: {
                "param": "up3x-no-denoise.param",
                "bin": "up3x-no-denoise.bin",
                "scale": 3,
                "folder": "models/CUGAN/models-se",
            },
            19: {
                "param": "up3x-denoise3x.param",
                "bin": "up3x-denoise3x.bin",
                "scale": 3,
                "folder": "models/CUGAN/models-se",
            },
            20: {
                "param": "up4x-conservative.param",
                "bin": "up4x-conservative.bin",
                "scale": 4,
                "folder": "models/CUGAN/models-se",
            },
            21: {
                "param": "up4x-no-denoise.param",
                "bin": "up4x-no-denoise.bin",
                "scale": 4,
                "folder": "models/CUGAN/models-se",
            },
            22: {
                "param": "up4x-denoise3x.param",
                "bin": "up3x-denoise3x.bin",
                "scale": 4,
                "folder": "models/CUGAN/models-se",
            },
            # cugan-pro models
            23: {
                "param": "up2x-denoise3x.param",
                "bin": "up2x-denoise3x.bin",
                "scale": 2,
                "folder": "models/CUGAN/models-pro",
            },
            24: {
                "param": "up2x-conservative.param",
                "bin": "up2x-conservative.bin",
                "scale": 2,
                "folder": "models/CUGAN/models-pro",
            },
            25: {
                "param": "up2x-no-denoise.param",
                "bin": "up2x-no-denoise.bin",
                "scale": 2,
                "folder": "models/CUGAN/models-pro",
            },
            26: {
                "param": "up3x-denoise3x",
                "bin": "denoise3x-up3x",
                "scale": 3,
                "folder": "models/CUGAN/models-pro",
            },
            27: {
                "param": "up3x-conservative",
                "bin": "up3x-conservative.bin",
                "scale": 3,
                "folder": "models/CUGAN/models-pro",
            },
            28: {
                "param": "up3x-no-denoise.param",
                "bin": "up3x-no-denoise.bin",
                "scale": 3,
                "folder": "models/CUGAN/models-pro",
            },
            # shufflecugan
            29: {
                "param": "sudo_shuffle_cugan-x2.param",
                "bin": "sudo_shuffle_cugan-x2.bin",
                "scale": 2,
                "folder": "models/SHUFFLECUGAN",
            },
        }

        if self._model == -1:
            if param_path is None and model_path is None and scale == 0:
                raise ValueError(
                    "param_path, model_path and scale must be specified when model == -1"
                )
            if param_path is None or model_path is None:
                raise ValueError(
                    "param_path and model_path must be specified when model == -1"
                )
            if scale == 0:
                raise ValueError("scale must be specified when model == -1")
        else:
            if self._model_str == "":
                model_dir = pathlib.Path(__file__).parent / model_dict[self._model].get(
                    "folder", "models"
                )

                param_path = model_dir / pathlib.Path(
                    str(model_dict[self._model]["param"])
                )
                model_path = model_dir / pathlib.Path(
                    str(model_dict[self._model]["bin"])
                )
            else:
                model_dir = pathlib.Path(self._model_str).parent

                param_path = model_dir / pathlib.Path(
                    str(self._model_str.split("/")[-1] + ".param")
                )
                model_path = model_dir / pathlib.Path(
                    str(self._model_str.split("/")[-1] + ".bin")
                )

                # print (model_dir,param_path,model_path)
        self._scale = scale if scale != 0 else int(model_dict[self._model]["scale"])
        self._set_parameters()

        if param_path is None or model_path is None:
            raise ValueError("param_path and model_path is None")

        self._upscale_object.load(str(param_path), str(model_path))

    def process(self) -> None:
        self._upscale_object.process(self.raw_in_image, self.raw_out_image)

    def process_cv2(self, _image: np.ndarray) -> np.ndarray:

        in_bytes = _image.tobytes()
        if self.channels == None:
            self.channels = int(len(in_bytes) / (_image.shape[1] * _image.shape[0]))
            self.out_bytes = (self._scale**2) * len(in_bytes) * b"\x00"

        self.raw_in_image = wrapped.UPSCALEImage(
            in_bytes, _image.shape[1], _image.shape[0], self.channels
        )

        self.raw_out_image = wrapped.UPSCALEImage(
            self.out_bytes,
            self._scale * _image.shape[1],
            self._scale * _image.shape[0],
            self.channels,
        )

        self.process()

        return np.frombuffer(self.raw_out_image.get_data(), dtype=np.uint8).reshape(
            self._scale * _image.shape[0], self._scale * _image.shape[1], self.channels
        )

    def process_bytes(
        self, _image_bytes: bytes, width: int, height: int, channels: int
    ) -> bytes:
        if self.raw_in_image is None and self.raw_out_image is None:
            self.raw_in_image = wrapped.UPSCALEImage(
                _image_bytes, width, height, channels
            )

            self.raw_out_image = wrapped.UPSCALEImage(
                (self._scale**2) * len(_image_bytes) * b"\x00",
                self._scale * width,
                self._scale * height,
                channels,
            )

        self.raw_in_image.set_data(_image_bytes)

        self.process()

        return self.raw_out_image.get_data()


class UpscaleWithNCNNMode:
    def __init__(
        self,
        modelPath: os.PathLike,
        num_threads: int,
        scale: int,
        gpuid: int = 0,
        width: int = 1920,
        height: int = 1080,
        tilesize: int = 0,
        tilePad=10,
        hdr_mode=False,
    ):
        self.tilewidth = width
        self.tileheight = height
        self.tile_size = tilesize if tilesize > 0 else 512
        self.tile_pad = tilePad
        self.scale = scale
        self.hdr_mode = hdr_mode

        self.net = ncnn.Net()
        # Use vulkan compute
        self.net.opt.use_vulkan_compute = True
        self.net.opt.use_fp16_packed = True
        self.net.opt.use_fp16_storage = True
        self.net.opt.use_fp16_arithmetic = False
        self.net.opt.use_int8_storage = True
        self.net.opt.use_int8_arithmetic = False
        self.net.set_vulkan_device(gpuid)
        self.blob_vkallocator = ncnn.VkBlobAllocator(self.net.vulkan_device())
        self.staging_vkallocator = ncnn.VkStagingAllocator(self.net.vulkan_device())
        self.net.opt.blob_vkallocator = self.blob_vkallocator
        self.net.opt.staging_vkallocator = self.staging_vkallocator
        self.net.opt.workspace_vkallocator = self.blob_vkallocator
        # Load model param and bin
        self.net.load_param(modelPath + ".param")
        self.net.load_model(modelPath + ".bin")

    def NCNNImageMatFromNP(self, npArray: np.array):
        return ncnn.Mat.from_pixels(
            npArray,
            ncnn.Mat.PixelType.PIXEL_BGR,
            self.tilewidth,
            self.tileheight,
        )

    def NormalizeImage(self, mat, norm_vals):
        mean_vals = []
        mat.substract_mean_normalize(mean_vals, norm_vals)

    def ClampNPArray(self, nparray: np.array) -> np.array:
        nparray = np.clip(nparray, 0, 255)
        return nparray

    def process_bytes(self, frame: bytes, *args, **kwargs) -> bytes:
        frame = np.ascontiguousarray(np.frombuffer(frame, dtype=np.uint8))
        ex = self.net.create_extractor()

        # frame = self.ClampNPArray(frame)
        frame = self.NCNNImageMatFromNP(frame)
        # norm
        self.NormalizeImage(mat=frame, norm_vals=[1 / 255.0, 1 / 255.0, 1 / 255.0])
        # render frame

        ex.input("data", frame)

        ret, frame = ex.extract("output")

        # norm
        frame = np.array(frame)
        frame = frame.transpose(1, 2, 0) * 255
        # frame = self.ClampNPArray(frame)
        return np.ascontiguousarray(frame, dtype=np.uint8).tobytes()

    def renderTiledImage(self, img: np.ndarray):
        raise NotImplementedError(
            "Tile rendering not implemented for default ncnn fallback, please install vcredlist from https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170"
        )
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.

        Modified from: https://github.com/ata4/esrgan-launcher
        """
        img = img.reshape(3, self.height, self.width)
        channel, height, width = img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (channel, output_height, output_width)

        # start with black image
        self.output = np.zeros(output_shape, dtype=img.dtype)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = img[
                    :,
                    input_start_y_pad:input_end_y_pad,
                    input_start_x_pad:input_end_x_pad,
                ]

                # upscale tile
                output_tile = self.procNCNNVk(input_tile).transpose(2, 0, 1)
                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[
                    :, output_start_y:output_end_y, output_start_x:output_end_x
                ] = output_tile[
                    :,
                    output_start_y_tile:output_end_y_tile,
                    output_start_x_tile:output_end_x_tile,
                ]
        return self.output


class UpscaleNCNN:
    def __init__(
        self,
        modelPath: os.PathLike,
        num_threads: int,
        scale: int,
        gpuid: int = 0,
        width: int = 1920,
        height: int = 1080,
        tilesize: int = 0,
        tilePad=10,
        hdr_mode=False,
    ):
        # only import if necessary
        self.pad_w = tilePad
        self.pad_h = tilePad
        self.gpuid = gpuid
        self.modelPath = modelPath
        self.scale = scale
        self.tilesize = tilesize
        self.width = width
        self.height = height
        self.tilewidth = width
        self.tileheight = height
        if tilesize != 0:
            self.tilewidth = tilesize
            self.tile_size = tilesize
            self.tileheight = tilesize
        self.scale = scale
        self.threads = num_threads
        self.tilePad = tilePad
        self.tile_pad = tilePad
        self.hdr_mode = hdr_mode
        self.backend = "ncnn"
        self.mean_vals = []
        self.norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
        self._load()

    def _load(self):
        from ..utils.Util import suppress_stdout_stderr

        with suppress_stdout_stderr():
            if method == "ncnn_vulkan":
                self.net = UpscaleWithNCNNMode(
                    modelPath=self.modelPath,
                    num_threads=self.threads,
                    scale=self.scale,
                    gpuid=self.gpuid,
                    width=self.width,
                    height=self.height,
                    tilesize=self.tilesize,
                    tilePad=self.tilePad,
                )
            elif method == "upscale_ncnn_py":
                self.net = UPSCALE(
                    gpuid=self.gpuid,
                    model_str=self.modelPath,
                    num_threads=self.threads,
                    scale=self.scale,
                    tilesize=self.tilesize,
                )
            device = ncnn.get_gpu_device(self.gpuid).info().device_name()
        print("Using GPU:", device)

    def hotUnload(self):
        self.model = None
        self.net = None

    def hotReload(self):
        self._load()

    def set_self_model(self):
        pass

    def frame_to_tensor(self, frame: np.array) -> np.array:
        return frame

    def __call__(self, imageChunk: Frame):
        while self.net is None:
            sleep(1)

        img = self.net.process_bytes(
            imageChunk.get_frame_bytes(), self.width, self.height, 3
        )
        retFrame = Frame(
            self.backend,
            self.width,
            self.height,
            imageChunk.device,
            gpu_id=imageChunk.gpu_id,
            hdr_mode=self.hdr_mode,
            dtype=imageChunk.dtype,
        )
        retFrame.set_frame_bytes(img)
        return retFrame
