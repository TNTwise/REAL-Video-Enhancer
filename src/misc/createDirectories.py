import os
from src.programData.thisdir import thisdir
homedir = os.path.expanduser(r"~")


def createDirectories():
    
    os.makedirs(
        os.path.join(f"{thisdir()}"),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(f"{thisdir()}", "files"),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(f"{thisdir()}", "bin"),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(f"{thisdir()}", "models"),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(f"{thisdir()}", "logs"),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(f"{thisdir()}", "models"),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(f"{thisdir()}", "renders"),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(f"{thisdir()}", "models", "custom_models_ncnn", "models"),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(f"{thisdir()}", "models", "custom-models-cuda"), exist_ok=True
    )

    os.makedirs(
        os.path.join(f"{thisdir()}", "models", "custom-models-cuda"), exist_ok=True
    )

    os.makedirs(
        os.path.join(f"{thisdir()}", "models", "tensorrt-engines"), exist_ok=True
    )
    os.makedirs(os.path.join(f"{thisdir()}", "models", "onnx-models"), exist_ok=True)


def createFiles():
    try:
        os.mknod(os.path.join(f"{thisdir()}", "files", "settings.txt"))
    except:
        pass
    os.system(f'touch {os.path.join(f"{thisdir()}","files","settings.txt")}')
