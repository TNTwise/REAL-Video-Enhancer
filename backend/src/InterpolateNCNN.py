from rife_ncnn_vulkan_python import Rife


class InterpolateRIFENCNN:
    def __init__(
        self,
        interpolateModelPath: str,
        width: int = 1920,
        height: int = 1080,
        threads: int = 1,
        gpuid: int = 0,
    ):
        self.interpolateModelPath = interpolateModelPath
        self.width = width
        self.height = height
        self.render = Rife(
            gpuid=gpuid,
            num_threads=threads,
            model=self.interpolateModelPath,
            uhd_mode=False,
            channels=3,
            height=height,
            width=width,
        )

    def process(self, img0, img1, timestep) -> bytes:
        return bytes(self.render.process_bytes(img0, img1, timestep))

    def bytesToByteArray(self, bytes):
        return bytearray(bytes)
