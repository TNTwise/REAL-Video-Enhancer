from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routers import (
    backend_router,
    install_packages_router,
    models_router,
    render_router,
    settings_router,
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
app.include_router(video_info_router)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
