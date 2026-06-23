from .install_packages_router import router as install_packages_router
from .render_router import router as render_router
from .settings_router import router as settings_router
from .video_info_router import router as video_info_router

__all__ = [
    "settings_router",
    "render_router",
    "video_info_router",
    "install_packages_router",
]
