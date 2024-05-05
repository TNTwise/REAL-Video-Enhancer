import os
import requests
import sys

import src.programData.thisdir

thisdir = src.programData.thisdir.thisdir()
from PyQt5.QtWidgets import QMessageBox
from src.programData.settings import *
from src.programData.return_data import *
import math
import shutil
from src.misc.log import *
from src.programData.write_permisions import *
import traceback
try:
    import cupy
    import modules.GMFSSCUDA as GMFSSCUDA

    gmfss = True
except Exception as e:
    gmfss = False
try:
    import torch
    import torchvision
    import spandrel

    cuda = True
except:
    cuda = False

try:
    import tensorrt
    from torch_tensorrt.fx import LowerSetting

    tensorRT = True
except:
    tensorRT = False


def check_if_online(dont_check=False, url="https://raw.githubusercontent.com/"):
    """
    Checks if the system is connected to the internet.

    Args:
        dont_check (bool, optional): If True, won't show a message box when offline. Defaults to False.
        url (str, optional): URL to check for internet connectivity. Defaults to "https://raw.githubusercontent.com/".

    Returns:
        bool: True if online, False otherwise.
    """
    online = False
    try:
        requests.get(url)
        online = True
    except Exception as e:
        traceback_info = traceback.format_exc()
        log(f"ERROR: {e} {traceback_info}")
        log(f"{e}")
        if not dont_check:
            msg = QMessageBox()
            msg.setWindowTitle(" ")
            msg.setText(
                f"You are offline, please connect to the internet to download the models."
            )
            msg.exec_()
        pass

    return online


def check_if_free_space(RenderDir):
    """
    Checks the free space on the disk.

    Args:
        RenderDir (str): Path to the directory to check for free space.

    Returns:
        int: Amount of free space in bytes.
    """
    return shutil.disk_usage(f"{RenderDir}").free


def check_if_enough_space_for_install(size_in_bytes):
    """
    Checks if there's enough free space on the disk for installation.

    Args:
        size_in_bytes (int): Size of the data to be installed in bytes.

    Returns:
        bool: True if enough space, False otherwise.
    """
    free_space = shutil.disk_usage(f"{thisdir}").free
    return free_space > size_in_bytes


def check_if_flatpak():
    """
    Checks if the application is running within Flatpak environment.

    Returns:
        bool: True if running in Flatpak, False otherwise.
    """
    if "FLATPAK_ID" in os.environ:
        return True
    return False


def check_if_enough_space_output_disk(input_file, render, times):
    """
    Checks if there's enough space on the output disk for processing.

    Args:
        input_file (str): Path to the input file.
        render (str): Type of rendering.
        times (int): Interpolation factor.

    Returns:
        tuple: A tuple containing a boolean indicating if enough space, required size, and available space in GB.
    """
    settings = Settings()
    img_type = settings.Image_Type
    if img_type == ".jpg":
        multiplier = 1 / 10
    if img_type == ".webp":
        multiplier = 1 / 30
    if img_type == ".png":
        multiplier = 2.34 / 3
    resolution = VideoName.return_video_resolution(input_file)
    frame_count = VideoName.return_video_frame_count(input_file)
    resolution_multiplier = math.ceil(resolution[1] * resolution[0])

    full_extraction_size = resolution_multiplier * frame_count * multiplier
    free_space = check_if_free_space(settings.OutputDir)
    if settings.RenderType == "Classic":
        if render == "esrgan":
            rnd = round(resolution * 0.001)
            full_size = full_extraction_size * times * rnd
            return full_size < free_space, full_size / (1024**3), free_space / (1024**3)

        if render == "rife":
            full_size = full_extraction_size * times
            return full_size < free_space, full_size / (1024**3), free_space / (1024**3)
    else:
        return (
            full_extraction_size * 5 < free_space,
            full_extraction_size * 5 / (1024**3),
            free_space / (1024**3),
        )


def check_if_enough_space(input_file, render, times):
    """
    Checks if there's enough space on the rendering disk for processing.

    Args:
        input_file (str): Path to the input file.
        render (str): Type of rendering.
        times (int): Interpolation factor.

    Returns:
        tuple: A tuple containing a boolean indicating if enough space, required size, and available space in GB.
    """
    settings = Settings()
    img_type = settings.Image_Type
    if img_type == ".jpg":
        multiplier = 1 / 10
    if img_type == ".webp":
        multiplier = 1 / 30
    if img_type == ".png":
        multiplier = 2.34 / 3
    resolution = VideoName.return_video_resolution(input_file)
    frame_count = VideoName.return_video_frame_count(input_file)
    resolution_multiplier = math.ceil(resolution[1] * resolution[0])

    full_extraction_size = resolution_multiplier * frame_count * multiplier
    free_space = check_if_free_space(settings.RenderDir)
    if settings.RenderType == "Classic":
        if render == "esrgan":
            rnd = round(resolution * 0.001)
            if rnd < 1:
                rnd = 1
            if img_type == ".png":
                full_size = (
                    full_extraction_size * (2) + full_extraction_size * times * rnd
                )

            if img_type == ".jpg":
                full_size = (
                    full_extraction_size * (2) + full_extraction_size * times * 5 * rnd
                )
            if img_type == ".webp":
                full_size = (
                    full_extraction_size * (2) + full_extraction_size * times * 4 * rnd
                )
            return full_size < free_space, full_size / (1024**3), free_space / (1024**3)

        if render == "rife":
            if img_type == ".png":
                full_size = full_extraction_size * (2) + full_extraction_size * times

            if img_type == ".jpg":
                full_size = (
                    full_extraction_size * (2) + full_extraction_size * times * 5
                )
            if img_type == ".webp":
                full_size = (
                    full_extraction_size * (2) + full_extraction_size * times * 4
                )
            return full_size < free_space, full_size / (1024**3), free_space / (1024**3)
    else:
        return (
            full_extraction_size * 5 < free_space,
            full_extraction_size * 5 / (1024**3),
            free_space / (1024**3),
        )


def isCUDA():
    return cuda


def isTensorRT():
    return tensorRT


def isCUPY():
    return gmfss


def check_for_individual_models():
    """
    Checks for individual models in the program directory.

    Returns:
        list: List of available models.
    """
    return_list = []
    if os.path.exists(f"{thisdir}/models/"):
        if os.path.exists(f"{thisdir}/models/rife/"):
            return_list.append("Rife-ncnn")
        if os.path.exists(f"{thisdir}/models/realesrgan/"):
            return_list.append("RealESRGAN-ncnn")
        if os.path.exists(f"{thisdir}/models/waifu2x/"):
            return_list.append("Waifu2X-ncnn")
        if os.path.exists(f"{thisdir}/models/realcugan/"):
            return_list.append("RealCUGAN-ncnn")
        if os.path.exists(f"{thisdir}/models/ifrnet/"):
            return_list.append("IFRNET-ncnn")
        if os.path.exists(f"{thisdir}/models/realsr/"):
            return_list.append("RealSR-ncnn")
        if os.path.exists(f"{thisdir}/models/vapoursynth-rife/"):
            return_list.append("Vapoursynth-RIFE")
        if (
            os.path.exists(f"{thisdir}/models/custom_models_ncnn/")
            and len(
                os.listdir(
                    os.path.join(f"{thisdir}", "models", "custom_models_ncnn", "models")
                )
            )
            > 0
        ):
            return_list.append("Custom NCNN Models")
        if os.path.exists(f"{thisdir}/models/span"):
            return_list.append("SPAN (NCNN)")
        if isCUDA():
            if os.path.exists(f"{thisdir}/models/rife-cuda/"):
                return_list.append("rife-cuda")
                if isTensorRT():
                    return_list.append("rife-cuda-trt")

            if os.path.exists(f"{thisdir}/models/realesrgan-cuda/"):
                return_list.append("realesrgan-cuda")
            if (
                os.path.exists(f"{thisdir}/models/custom-models-cuda/")
                and len(os.listdir(f"{thisdir}/models/custom-models-cuda/")) > 0
            ):
                return_list.append("custom-cuda-models")

            if isCUPY():
                if os.path.exists(f"{thisdir}/models/gfmss-cuda/"):
                    return_list.append("gfmss-cuda")

        if len(return_list) > 0:
            return return_list

    return None


def check_for_each_binary():
    """
    Checks for required binaries in the program directory.

    Returns:
        bool: True if all binaries exist, False otherwise.
    """
    if (
        os.path.isfile(f"{thisdir}/bin/ffmpeg")
        and os.path.isfile(f"{thisdir}/bin/yt-dlp_linux")
        and os.path.isfile(f"{thisdir}/bin/glxinfo")
    ):
        return True
    return False


