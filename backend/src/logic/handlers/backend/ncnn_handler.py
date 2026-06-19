from .backend_handler import BackendHandler
from src.utils.LogConfig import get_logger

logger = get_logger(__name__)

class NCNNHandler(BackendHandler)
    def __init__(self):
        try:
            import ncnn
            import rife_ncnn_vulkan_python
            import upscale_ncnn_py
            self._ncnn = ncnn
            self._rife = rife_ncnn_vulkan_python
            self._upscale = upscale_ncnn_py
        except ImportError:
            self._ncnn = None
            self._rife = None
            self._upscale = None
            logger.error("NCNN Not installed!")

    def is_available(self) -> bool:
        return self._ncnn is not None and self._rife is not None and self._upscale is not None

    def get_ncnn(self):
        # TODO: Custom exception
        if not self._ncnn:
            raise Exception("NCNN Does not exist! Cannot get backend that is not installed")
        return self._ncnn

    def get_rife(self):
        if not self._rife:
            raise Exception("RIFE NCNN does not exist!")
        return self._rife

    def get_upsale(self):
        if not self._upscale:
            raise Exception("Upscale NCNN Not found!")
        return self._upscale
