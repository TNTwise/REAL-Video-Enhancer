from rife_ncnn_vulkan_python import Rife
from time import sleep


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
        self.gpuid = gpuid
        self.threads = threads
        self._load()

    def _load(self):
        self.render = Rife(
            gpuid=self.gpuid,
            num_threads=self.threads,
            model=self.interpolateModelPath,
            uhd_mode=False,
            channels=3,
            height=self.height,
            width=self.width,
        )

    def hotUnload(self):
        self.render = None

    def hotReload(self):
        self._load()

    def process(self, img0, img1, timestep) -> bytes:
        while self.render is None:
            sleep(1)
        frame = self.render.process_bytes(img0, img1, timestep)
        return frame

    def uncacheFrame(self):
        self.render.uncache_frame()
