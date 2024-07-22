

import os
import subprocess
import argparse


def pipInstall(file):
    command = ["python3", "-m", "pip", "install", "-r"]
    command.append(file)
    subprocess.run(command)


def installImportLib():
    command = ["python3", "-m", "pip", "install", "--upgrade", "pip", "setuptools"]
    subprocess.run(command)
    command = ["python3", "-m", "pip", "install", "importlib"]
    subprocess.run(command)


def buildenv():
    filelist = [
        "mainwindow",
        os.path.join("src", "getModels", "SelectModels"),
        os.path.join("src", "getModels", "Download"),
        os.path.join("src", "getModels", "SelectAI"),
        os.path.join("src", "getLinkVideo", "get_vid_from_link"),
    ]
    for file in filelist:
        with open(f"{file}.py", "w") as f:
            uic.compileUi(f"{file}.ui", f)


def buildNCNNLinux():
    """
    Builds REAL Video Enhancer with only ncnn, needs python 3.8
    """
    pipInstall("requirements-NCNN-Linux.txt")
    command = ["python3", "-m", "cx_Freeze", "-c", "main.py", "--target-dir", "dist"]
    subprocess.run(command)


def fixEvalFrame():
    with open(
        os.path.join("dist", "main", "_internal", "torch", "_dynamo", "eval_frame.py"),
        "w",
    ) as f:
        f.write(fixedEvalFrame)


def buildCUDALinux():
    """
    Builds REAL Video Enhancer with cuda, needs python 3.11
    """
    pipInstall("requirements-CUDA-Linux.txt")
    installImportLib()
    command = [
        "python3",
        "-m",
        "PyInstaller",
        "main.py",
        "--collect-all",
        "nvidia",
        "--collect-all",
        "polygraphy",
        "--collect-all",
        "torch",
        "--collect-all",
        "torch._C",
        "--collect-all",
        "tensorrt",
        "--collect-all",
        "tensorrt-cu12-bindings",
        "--collect-all",
        "tensorrt_libs",
        "--collect-all",
        "pytorch-triton",
        "--collect-all",
        "triton",
        "--hidden-import",
        "upscale_ncnn_py.upscale_ncnn_py_wrapper",
        "--hidden-import",
        "realcugan_ncnn_py.realcugan_ncnn_py_wrapper",
        "--hidden-import",
        "rife_ncnn_vulkan_python.rife_ncnn_vulkan_wrapper",
    ]
    subprocess.run(command)


def buildROCmLinux():
    """
    Builds REAL Video Enhancer with rocm, needs python 3.11
    """

    pipInstall("requirements-ROCM-Linux.txt")
    installImportLib()
    command = [
        "python3",
        "-m",
        "PyInstaller",
        "main.py",
        "--collect-all",
        "torch",
        "--collect-all",
        "pytorch-triton",
        "--collect-all",
        "triton",
        "--hidden-import",
        "upscale_ncnn_py.upscale_ncnn_py_wrapper",
        "--hidden-import",
        "realcugan_ncnn_py.realcugan_ncnn_py_wrapper",
        "--hidden-import",
        "rife_ncnn_vulkan_python.rife_ncnn_vulkan_wrapper",
    ]
    subprocess.run(command)


def buildNCNNFlatpakLinux():
    pass


def buildNCNNMacOS():
    command = [
        "python3",
        "-m",
        "PyInstaller",
        "--onefile",
        "main.py",
        "--hidden-import",
        "upscale_ncnn_py.upscale_ncnn_py_wrapper",
        "--hidden-import",
        "realcugan_ncnn_py.realcugan_ncnn_py_wrapper",
        "--hidden-import",
        "rife_ncnn_vulkan_python.rife_ncnn_vulkan_wrapper",
    ]
    subprocess.run(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RVE")
    parser.add_argument(
        "--build_cuda",
        help="builds cuda rve, requires python 3.10",
        action="store_true",
    )
    parser.add_argument(
        "--build_ncnn",
        help="builds ncnn rve, requires python 3.8",
        action="store_true",
    )
    parser.add_argument(
        "--build_rocm",
        help="builds rocm rve, requires python 3.11",
        action="store_true",
    )
    parser.add_argument(
        "--build_mac_ncnn",
        help="builds macos ncnn rve, requires python 3.11",
        action="store_true",
    )

    args = parser.parse_args()

    pipInstall("requirements.txt")
    import PyQt5.uic as uic

    buildenv()
    if args.build_mac_ncnn:
        buildNCNNMacOS()
    if args.build_rocm:
        buildROCmLinux()
    if args.build_cuda:
        buildCUDALinux()
        #fixEvalFrame()
    if args.build_ncnn:
        buildNCNNLinux()
