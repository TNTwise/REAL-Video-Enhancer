import os
import src.programData.thisdir

thisdir = src.programData.thisdir.thisdir()
import src.programData.checks as checks
import src.programData.return_data as return_data


def returnCorrectLinkBasedOnOS(link):
    if return_data.returnOperatingSystem() == "Linux":
        return link

    if return_data.returnOperatingSystem() == "MacOS":
        if "rife-ncnn-vulkan" in link:
            return link.replace("rife-ncnn-vulkan", "rife-ncnn-vulkan-macos-bin")
        return link.replace("-ubuntu", "-macos")


def cudaRifeModels(self, install_modules_dict: dict = {}):
    modelDict = {}
    items=["rife4.13-lite.pkl","rife4.14.pkl","rife4.14-lite.pkl","rife4.15.pkl","rife4.16-lite.pkl"]
    if (
        self.ui.rife46CUDA.isChecked()
        
    ):
        if os.path.exists(f"{thisdir}/models/rife-cuda/rife46") == False:
            modelDict[
            "https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/rife4.6.pkl"
        ] = "rife4.6.pkl"
        items.append("rife4.6.pkl")
    
    if (
        self.ui.rife413liteCUDA.isChecked()
        
    ):
        if os.path.exists(f"{thisdir}/models/rife-cuda/rife413-lite") == False:
            modelDict[
            "https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/rife4.13-lite.pkl"
            ] = "rife4.13-lite.pkl"
        items.append("rife4.13-lite.pkl")
    if (
        self.ui.rife414CUDA.isChecked()
        
    ):  
        if os.path.exists(f"{thisdir}/models/rife-cuda/rife414") == False:
            modelDict[
                "https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/rife4.14.pkl"
            ] = "rife4.14.pkl"
        items.append("rife4.14.pkl")

    if (
        self.ui.rife414liteCUDA.isChecked()
    ):
        if os.path.exists(f"{thisdir}/models/rife-cuda/rife414-lite") == False:
            modelDict[
            "https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/rife4.14-lite.pkl"
            ] = "rife4.14-lite.pkl"
        items.append("rife4.14-lite.pkl")
    if (
        self.ui.rife415CUDA.isChecked()
    ):
        if os.path.exists(f"{thisdir}/models/rife-cuda/rife415") == False:
            modelDict[
            "https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/rife4.15.pkl"
            ] = "rife4.15.pkl"
        items.append("rife4.15.pkl")
    if (
        self.ui.rife416liteCUDA.isChecked()
        
    ):
        if os.path.exists(f"{thisdir}/models/rife-cuda/rife416-lite") == False:
            modelDict[
            "https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/rife4.16-lite.pkl"
            ] = "rife4.16-lite.pkl"
        items.append("rife4.16-lite.pkl")
        
    # remove unwanted models
    
    items2 = []
    for item in items:
        items2.append(item.replace(".pkl","").replace(".",""))
    for i in os.listdir(f'{thisdir}/models/rife-cuda/'):
        if i not in items2:
            os.system(f'rm -rf "{thisdir}/models/rife-cuda/{i}"')
            
    install_modules_dict.update(modelDict)
def returnModelList(
    self, settings
):  # make sure names match up on both selectAI.ui and main.ui
    rife_install_list = []
    if os.path.exists(f"{settings.ModelDir}/") == False:
        os.mkdir(f"{settings.ModelDir}/")
    try:
        if self.ui.RifeCheckBox.isChecked():
            with open(f"{thisdir}/models.txt", "r") as f:
                for i in f.readlines():
                    i = i.replace("\n", "")
                    rife_install_list.append(i)
                    if "v4" in i:
                        rife_install_list.append(f"{i}-ensemble")

    except Exception as e:
        if self.ui.RifeCheckBox.isChecked():
            rife_install_list.append("rife-v4.6")

    install_modules_dict = {}

    if (
        self.ui.RealSRCheckBox.isChecked()
        and os.path.exists(f"{settings.ModelDir}/realsr/") == False
    ):
        install_modules_dict[
            returnCorrectLinkBasedOnOS(
                "https://github.com/nihui/realsr-ncnn-vulkan/releases/download/20220728/realsr-ncnn-vulkan-20220728-ubuntu.zip"
            )
        ] = returnCorrectLinkBasedOnOS("realsr-ncnn-vulkan-20220728-ubuntu.zip")
    if (
        self.ui.CustomModelsNCNNCheckBox.isChecked()
        and os.path.exists(f"{settings.ModelDir}/custom_models_ncnn/") == False
    ):
        install_modules_dict[
            returnCorrectLinkBasedOnOS(
                "https://github.com/TNTwise/Universal-NCNN-Upscaler/releases/download/20240218/upscayl-custom_models_ncnn-ncnn-vulkan-20240218-ubuntu.zip"
            )
        ] = returnCorrectLinkBasedOnOS(
            "upscayl-custom_models_ncnn-ncnn-vulkan-20240218-ubuntu.zip"
        )
    if (
        self.ui.RifeCheckBox.isChecked() == True
        and os.path.exists(f"{settings.ModelDir}/rife/") == False
    ):
        install_modules_dict[
            returnCorrectLinkBasedOnOS(
                "https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/rife-ncnn-vulkan"
            )
        ] = "rife-ncnn-vulkan"

    if (
        self.ui.RealESRGANCheckBox.isChecked() == True
        and os.path.exists(f"{settings.ModelDir}/realesrgan") == False
    ):
        install_modules_dict[
            returnCorrectLinkBasedOnOS(
                "https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
            )
        ] = returnCorrectLinkBasedOnOS("realesrgan-ncnn-vulkan-20220424-ubuntu.zip")

    if (
        self.ui.Waifu2xCheckBox.isChecked() == True
        and os.path.exists(f"{settings.ModelDir}/waifu2x") == False
    ):
        install_modules_dict[
            returnCorrectLinkBasedOnOS(
                "https://github.com/nihui/waifu2x-ncnn-vulkan/releases/download/20220728/waifu2x-ncnn-vulkan-20220728-ubuntu.zip"
            )
        ] = returnCorrectLinkBasedOnOS("waifu2x-ncnn-vulkan-20220728-ubuntu.zip")

    if (
        self.ui.CainCheckBox.isChecked() == True
        and os.path.exists(f"{settings.ModelDir}/ifrnet") == False
    ):
        install_modules_dict[
            returnCorrectLinkBasedOnOS(
                "https://github.com/nihui/ifrnet-ncnn-vulkan/releases/download/20220720/ifrnet-ncnn-vulkan-20220720-ubuntu.zip"
            )
        ] = returnCorrectLinkBasedOnOS("ifrnet-ncnn-vulkan-20220720-ubuntu.zip")

    if (
        self.ui.RealCUGANCheckBox.isChecked() == True
        and os.path.exists(f"{settings.ModelDir}/realcugan") == False
    ):
        install_modules_dict[
            returnCorrectLinkBasedOnOS(
                "https://github.com/nihui/realcugan-ncnn-vulkan/releases/download/20220728/realcugan-ncnn-vulkan-20220728-ubuntu.zip"
            )
        ] = returnCorrectLinkBasedOnOS("realcugan-ncnn-vulkan-20220728-ubuntu.zip")
    if (
        self.ui.RifeCUDACheckBox.isChecked() == True
        and os.path.exists(f"{settings.ModelDir}/rife-cuda") == False
    ):
        install_modules_dict[
            "https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/rife4.15.pkl"
        ] = "rife4.15.pkl"
    if (
        self.ui.RealESRGANCUDACheckBox.isChecked() == True
        and os.path.exists(f"{settings.ModelDir}/realesrgan-cuda") == False
    ):
        install_modules_dict[
            "https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/realesrgan-cuda.zip"
        ] = "realesrgan-cuda.zip"
    for i in rife_install_list:
        if (
            os.path.exists(f"{settings.ModelDir}/rife/rife-ncnn-vulkan") == False
            or checks.check_for_updated_binary("rife-ncnn-vulkan") == False
        ):
            install_modules_dict[
                returnCorrectLinkBasedOnOS(
                    "https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/rife-ncnn-vulkan"
                )
            ] = "rife-ncnn-vulkan"
        if os.path.exists(f"{settings.ModelDir}/rife/{i}") == False:
            install_modules_dict[
                f"https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/{i}.tar.gz"
            ] = f"{i}.tar.gz"
    if (
        rife_install_list == []
        and self.ui.RifeCheckBox.isChecked()
        and os.path.exists(f"{settings.ModelDir}/rife") == False
    ):
        install_modules_dict[
            returnCorrectLinkBasedOnOS(
                "https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/rife-ncnn-vulkan"
            )
        ] = "rife-ncnn-vulkan"
        install_modules_dict[
            f"https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/rife-v4.6.tar.gz"
        ] = f"rife-v4.6.tar.gz"
        install_modules_dict[
            f"https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/rife-v4.6-ensemble.tar.gz"
        ] = f"rife-v4.6-ensemble.tar.gz"
    if len(install_modules_dict) == 0 and len(os.listdir(f"{settings.ModelDir}/")) == 0:
        install_modules_dict[
            returnCorrectLinkBasedOnOS(
                "https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/rife-ncnn-vulkan"
            )
        ] = "rife-ncnn-vulkan"
        install_modules_dict[
            f"https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/rife-v4.6.tar.gz"
        ] = f"rife-v4.6.tar.gz"
        install_modules_dict[
            f"https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/rife-v4.6-ensemble.tar.gz"
        ] = f"rife-v4.6-ensemble.tar.gz"
    if (
        os.path.isfile(f"{thisdir}/bin/ffmpeg")
        and os.path.isfile(f"{thisdir}/bin/glxinfo")
        and os.path.isfile(f"{thisdir}/bin/yt-dlp_linux")
    ):
        pass
    else:
        if return_data.returnOperatingSystem() == "Linux":
            install_modules_dict.update(
                {
                    "https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/glxinfo": "glxinfo",
                    "https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/ffmpeg": "ffmpeg",
                    "https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/yt-dlp_linux": "yt-dlp_linux",
                }
            )
        if return_data.returnOperatingSystem() == "MacOS":
            install_modules_dict.update(
                {
                    "https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/glxinfo": "glxinfo",
                    "https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/ffmpeg-macos-bin": "ffmpeg",
                    "https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/yt-dlp_macos": "yt-dlp_linux",
                }
            )
    cudaRifeModels(self, install_modules_dict)
    return install_modules_dict
