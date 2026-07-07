from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routers import (
    backend_router,
    install_packages_router,
    models_router,
    render_router,
    settings_router,
    system_router,
    video_info_router,
)

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(backend_router)
app.include_router(install_packages_router)
app.include_router(models_router)
app.include_router(settings_router)
app.include_router(render_router)
app.include_router(system_router)
app.include_router(video_info_router)
if __name__ == "__main__":
    import os
    import socket
    import uvicorn

    port = os.environ.get("REV_PORT")
    if port is not None:
        port = int(port)
    else:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

    print(f"BACKEND_PORT:{port}", flush=True)
    uvicorn.run(app, host=os.environ.get("REV_HOST", "127.0.0.1"), port=port)
