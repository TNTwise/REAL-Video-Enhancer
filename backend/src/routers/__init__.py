from .backend_router import router as backend_router
from .install_packages_router import router as install_packages_router
from .models_router import router as models_router
from .render_router import router as render_router
from .settings_router import router as settings_router
from .system_router import router as system_router
from .video_info_router import router as video_info_router

__all__ = [
    "backend_router",
    "models_router",
    "settings_router",
    "render_router",
    "video_info_router",
    "install_packages_router",
    "system_router",
]
