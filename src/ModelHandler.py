from .DownloadModels import DownloadModel
from .ui.QTcustom import NetworkCheckPopup

"""
Key value pairs of the model name in the GUI
Data inside the tuple:
[0] = file in models directory
[1] = file to download
[2] = upscale times
[3] = arch
"""
ncnnInterpolateModels = {
    "RIFE 4.6 (Fastest Model)": ("rife-v4.6", "rife-v4.6.tar.gz", 1, "rife46"),
    "RIFE 4.7 (Smoothest Model)": ("rife-v4.7", "rife-v4.7.tar.gz", 1, "rife47"),
    "RIFE 4.15": ("rife-v4.15", "rife-v4.15.tar.gz", 1, "rife413"),
    "RIFE 4.18 (Recommended for realistic scenes)": (
        "rife-v4.18",
        "rife-v4.18.tar.gz",
        1,
        "rife413",
    ),
    "RIFE 4.22 (Latest General Model)": (
        "rife-v4.22",
        "rife-v4.22.tar.gz",
        1,
        "rife421",
    ),
    "RIFE 4.22-lite (Recommended Model)": (
        "rife-v4.22-lite",
        "rife-v4.22-lite.tar.gz",
        1,
        "rife422-lite",
    ),
}
pytorchInterpolateModels = {
    "RIFE 4.6 (Fastest Model)": ("rife4.6.pkl", "rife4.6.pkl", 1, "rife46"),
    "RIFE 4.7 (Smoothest Model)": ("rife4.7.pkl", "rife4.7.pkl", 1, "rife47"),
    "RIFE 4.15": ("rife4.15.pkl", "rife4.15.pkl", 1, "rife413"),
    "RIFE 4.18 (Recommended for realistic scenes)": (
        "rife4.18.pkl",
        "rife4.18.pkl",
        1,
        "rife413",
    ),
    "RIFE 4.22 (Latest General Model)": ("rife4.22.pkl", "rife4.22.pkl", 1, "rife421"),
    "RIFE 4.22-lite (Recommended Model)": (
        "rife4.22-lite.pkl",
        "rife4.22-lite.pkl",
        1,
        "rife422-lite",
    ),
}
tensorrtInterpolateModels = {
    "RIFE 4.6 (Fastest Model)": ("rife4.6.pkl", "rife4.6.pkl", 1, "rife46"),
    "RIFE 4.7 (Smoothest Model)": ("rife4.7.pkl", "rife4.7.pkl", 1, "rife47"),
    "RIFE 4.15": ("rife4.15.pkl", "rife4.15.pkl", 1, "rife413"),
    "RIFE 4.18 (Recommended for realistic scenes)": (
        "rife4.18.pkl",
        "rife4.18.pkl",
        1,
        "rife413",
    ),
    "RIFE 4.22 (Latest General Model)": ("rife4.22.pkl", "rife4.22.pkl", 1, "rife421"),
    "RIFE 4.22-lite (Recommended Model)": (
        "rife4.22-lite.pkl",
        "rife4.22-lite.pkl",
        1,
        "rife422-lite",
    ),
}
ncnnUpscaleModels = {
    "SPAN (Animation) (2X) (Fast)": (
        "2x_ModernSpanimationV2",
        "2x_ModernSpanimationV2.tar.gz",
        2,
        "SPAN",
    ),
    "SPAN (Realistic) (High Quality Source) (4X) (Fast)": (
        "4xNomos8k_span_otf_weak",
        "4xNomos8k_span_otf_weak.tar.gz",
        4,
        "SPAN",
    ),
    "SPAN (Realistic) (Medium Quality Source) (4X) (Fast)": (
        "4xNomos8k_span_otf_medium",
        "4xNomos8k_span_otf_medium.tar.gz",
        4,
        "SPAN",
    ),
    "SPAN (Realistic) (Low Quality Source) (4X) (Fast)": (
        "4xNomos8k_span_otf_strong",
        "4xNomos8k_span_otf_strong.tar.gz",
        4,
        "SPAN",
    ),
    "Compact (Realistic) (HD Input) (2X) (Fast)": (
        "2x_OpenProteus_Compact_i2_70K",
        "2x_OpenProteus_Compact_i2_70K.tar.gz",
        2,
        "Compact",
    ),
}
""""RealCUGAN Pro (Animation) (2X) (Slow)": (
        "up2x-conservative",
        "up2x-conservative.tar.gz",
        2,
        "compact",
    ),
    "RealCUGAN Pro (Animation) (3X) (Slow)": (
        "up3x-conservative",
        "up2x-conservative.tar.gz",
        3,
        "compact",
    ),
    "RealESRGAN (Animation) (2X) (Fast)": (
        "realesr-animevideov3-x2",
        "realesr-animevideov3-x2.tar.gz",
        2,
        "compact",
    ),
    "RealESRGAN (Animation) (3X) (Fast)": (
        "realesr-animevideov3-x3",
        "realesr-animevideov3-x3.tar.gz",
        3,
        "compact",
    ),
    "RealESRGAN (Animation) (4X) (Fast)": (
        "realesr-animevideov3-x4",
        "realesr-animevideov3-x4.tar.gz",
        4,
        "compact",
    ),"""
""""RealESRGAN Plus (General Model) (4X) (Slow)": (
    "realesrgan-x4plus",
    "realesrgan-x4plus.tar.gz",
    4,
    "esrgan",
),
"RealESRGAN Plus (Animation Model) (4X) (Slow)": (
    "realesrgan-x4plus-anime",
    "realesrgan-x4plus-anime.tar.gz",
    4,
    "esrgan",
),"""
pytorchUpscaleModels = {
    "SPAN (Animation) (2X) (Fast)": (
        "2x_ModernSpanimationV2.pth",
        "2x_ModernSpanimationV2.pth",
        2,
        "SPAN",
    ),
    "Sudo Shuffle SPAN (Animation) (2X) (Fast)": (
        "2xSudoShuffleSPAN.pth",
        "2xSudoShuffleSPAN.pth",
        2,
        "SPAN",
    ),
    "SPAN (Realistic) (High Quality Source) (4X) (Fast)": (
        "4xNomos8k_span_otf_weak.pth",
        "4xNomos8k_span_otf_weak.pth",
        4,
        "SPAN",
    ),
    "SPAN (Realistic) (Medium Quality Source) (4X) (Fast)": (
        "4xNomos8k_span_otf_medium.pth",
        "4xNomos8k_span_otf_medium.pth",
        4,
        "SPAN",
    ),
    "SPAN (Realistic) (Low Quality Source) (4X) (Fast)": (
        "4xNomos8k_span_otf_strong.pth",
        "4xNomos8k_span_otf_strong.pth",
        4,
        "SPAN",
    ),
    "Compact (Realistic) (HD Input) (2X) (Fast)": (
        "2x_OpenProteus_Compact_i2_70K.pth",
        "2x_OpenProteus_Compact_i2_70K.pth",
        2,
        "Compact",
    ),
}
tensorrtUpscaleModels = {
    "SPAN (Animation) (2X) (Fast)": (
        "2x_ModernSpanimationV2.pth",
        "2x_ModernSpanimationV2.pth",
        2,
        "SPAN",
    ),
    """"Sudo Shuffle SPAN (Animation) (2X) (Fast)": (
        "2xSudoShuffleSPAN.pth",
        "2xSudoShuffleSPAN.pth",
        2,
        "SPAN",
    ),"""
    "SPAN (Realistic) (High Quality Source) (4X) (Fast)": (
        "4xNomos8k_span_otf_weak.pth",
        "4xNomos8k_span_otf_weak.pth",
        4,
        "SPAN",
    ),
    "SPAN (Realistic) (Medium Quality Source) (4X) (Fast)": (
        "4xNomos8k_span_otf_medium.pth",
        "4xNomos8k_span_otf_medium.pth",
        4,
        "SPAN",
    ),
    "SPAN (Realistic) (Low Quality Source) (4X) (Fast)": (
        "4xNomos8k_span_otf_strong.pth",
        "4xNomos8k_span_otf_strong.pth",
        4,
        "SPAN",
    ),
    "Compact (Realistic) (HD Input) (2X) (Fast)": (
        "2x_OpenProteus_Compact_i2_70K.pth",
        "2x_OpenProteus_Compact_i2_70K.pth",
        2,
        "Compact",
    ),
}
onnxInterpolateModels = {
    "RIFE 4.22 (Recommended Model)": (
        "rife422_v2_ensembleFalse_op20_clamp.onnx",
        "rife422_v2_ensembleFalse_op20_clamp.onnx",
        1,
        "rife422-lite",
    ),
}
onnxUpscaleModels = {
    "SPAN (Animation) (2X)": (
        "2x_ModernSpanimationV2_clamp_op20.onnx",
        "2x_ModernSpanimationV2_clamp_op20.onnx",
        2,
        "SPAN",
    ),
}

totalModels = (
    onnxInterpolateModels
    | onnxUpscaleModels
    | pytorchInterpolateModels
    | pytorchUpscaleModels
    | ncnnInterpolateModels
    | ncnnUpscaleModels
    | tensorrtInterpolateModels
    | tensorrtUpscaleModels
)  # this doesnt include all models due to overwriting, but includes every case of every unique model name


def downloadModelsBasedOnInstalledBackend(installed_backends: list):
    if NetworkCheckPopup():
        for backend in installed_backends:
            match backend:
                case "ncnn":
                    for model in ncnnInterpolateModels:
                        DownloadModel(model, ncnnInterpolateModels[model][1], "ncnn")
                    for model in ncnnUpscaleModels:
                        DownloadModel(model, ncnnUpscaleModels[model][1], "ncnn")
                case "pytorch":  # no need for tensorrt as it uses pytorch models
                    for model in pytorchInterpolateModels:
                        DownloadModel(
                            model, pytorchInterpolateModels[model][1], "pytorch"
                        )
                    for model in pytorchUpscaleModels:
                        DownloadModel(model, pytorchUpscaleModels[model][1], "pytorch")
                case "directml":
                    for model in onnxInterpolateModels:
                        DownloadModel(model, onnxInterpolateModels[model][1], "onnx")
                    for model in onnxUpscaleModels:
                        DownloadModel(model, onnxUpscaleModels[model][1], "onnx")
