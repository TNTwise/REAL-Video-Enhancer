import asyncio
from functools import lru_cache

from fastapi import APIRouter, Depends
from src.logic.proxy.settings import PersistentSettingsProxy
from src.logic.services.install_packages_service import InstallPackagesService

router = APIRouter(prefix="/install-packages", tags=["install packages"])


@lru_cache
def get_install_packages_service() -> InstallPackagesService:
    return InstallPackagesService(PersistentSettingsProxy())


@router.post("/torch")
async def install_torch(
    upgrade: bool = False,
    service: InstallPackagesService = Depends(get_install_packages_service),
):
    asyncio.create_task(service.install_torch_packages(upgrade=upgrade))
    return {"status": "started"}


@router.post("/ncnn")
async def install_ncnn(
    upgrade: bool = False,
    service: InstallPackagesService = Depends(get_install_packages_service),
):
    asyncio.create_task(service.install_ncnn_packages(upgrade=upgrade))
    return {"status": "started"}


@router.post("/tensorrt")
async def install_tensorrt(
    upgrade: bool = False,
    service: InstallPackagesService = Depends(get_install_packages_service),
):
    asyncio.create_task(service.install_tensorrt_packages(upgrade=upgrade))
    return {"status": "started"}


@router.post("/base")
async def install_base(
    upgrade: bool = False,
    service: InstallPackagesService = Depends(get_install_packages_service),
):
    asyncio.create_task(service.install_base_packages())
    return {"status": "started"}


@router.get("/stdout")
def get_stdout(
    service: InstallPackagesService = Depends(get_install_packages_service),
):
    return {"stdout": service.current_stdout}


@router.get("/status")
def get_status(
    service: InstallPackagesService = Depends(get_install_packages_service),
):
    return {"status": service.current_status}
