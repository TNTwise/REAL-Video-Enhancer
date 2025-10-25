# REAL Video Enhancer
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FTNTwise%2FREAL-Video-enhancer%2F&countColor=%23263759)
[![pypresence](https://img.shields.io/badge/using-pypresence-00bb88.svg?style=for-the-badge&logo=discord&logoWidth=20)](https://github.com/qwertyquerty/pypresence)

![license](https://img.shields.io/github/license/tntwise/real-video-enhancer)
![Version](https://img.shields.io/badge/Version-2.3.8-blue)
![downloads_total](https://img.shields.io/github/downloads/tntwise/REAL-Video-Enhancer/total.svg?label=downloads%40total)
<a href="https://discord.gg/hwGHXga8ck">
      <img src="https://img.shields.io/discord/1041502781808328704?label=Discord" alt="Discord Shield"/></a>
<br/>
<a href="https://flathub.org/apps/io.github.tntwise.REAL-Video-Enhancer">
    <img src="https://dl.flathub.org/assets/badges/flathub-badge-en.svg" height="50px"/>
  </a>


<p align=center>
  <img src="https://github.com/TNTwise/REAL-Video-Enhancer/raw/2.0/icons/logo-v2.svg" width = "25%">
</p>

# Table of Contents
  
* **[Introduction](#introduction)**
* **[Features](#Features)**
* **[Hardware Requirements](#hardware-requirements)**
* **[Models](#models)**
  * [Interpolate Models](#interpolate-models)
  * [Upscale Models](#upscale-models)
* **[Backends](#backends)**
* **[FAQ](#faq)**
  * [General App Usage](#general-application-usage) 
  * [TensorRT](#tensorrt-related-questions)
  * [ROCm](#rocm-related-questions)
  * [NCNN](#ncnn-related-questions)
* **[Cloning](#cloning)**
* **[Building](#building)**
* **[Colab Notebook](#colab-notebook)**
* **[Credits](#credits)**
  * [People](#people) 
  * [Software](#software)

# Introduction

<strong>REAL Video Enhancer</strong>  is a redesigned and enhanced version of the original Rife ESRGAN App for Linux. This program offers convenient access to frame interpolation and upscaling functionalities on Windows, Linux and MacOS, and is an alternative to outdated software like <a rel="noopener noreferrer" href="https://nmkd.itch.io/flowframes" target="_blank" >Flowframes</a> or <a rel="noopener noreferrer" href="https://github.com/mafiosnik777/enhancr" target="_blank">enhancr</a>.

<p align=center>
  <img src="https://github.com/TNTwise/REAL-Video-Enhancer/blob/v2-main/screenshots/demo.png?raw=true" width = "100%">
</p>
<h1>Features: </h1>
<ul>
  <li> Windows support. <strong>!!! NOTICE !!!</strong> The bin can be detected as a trojan. This is a false positive caused by pyinstaller.</li>
  <li> Ubuntu 22.04+ support on Executable and Flatpak. (20.04 can work but is now legacy) </li>
  <li> MacOS 14+ arm/x86 support </li>
  <li> Discord RPC support for Discord system package and Discord flatpak. </li>
  <li> Scene change detection to preserve sharp transitions. </li>
  <li> Preview that shows latest frame that has been rendered. </li>
  <li> TensorRT and NCNN for efficient inference across many GPUs. </li>
</ul>

# Hardware/Software Requirements
|  | Minimum | Recommended | 
 |--|--|--|
| CPU | Dual Core x64 bit | Quad Core x64 bit
| GPU | Vulkan 1.3 capable device | Nvidia RTX GPU (20 series and up)
| VRAM | 4 GB - NCNN | 8 GB - TensorRT (Nvidia, why keep making 8gb cards?)
| RAM | 16 GB | 32 GB
| Storage | 1 GB free - NCNN | 16 GB free - TensorRT
| Operating System | Windows 10/11 64bit / MacOS 14+ | Any modern Linux distro (Ubuntu 22.04+)

# Models:
### Interpolate Models:
| Model | Author | Link |
|--|--|--|
| RIFE 4.6,4.7,4.15,4.18,4.22,4.22-lite,4.25 | Hzwer | [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) 
| GMFSS | 98mxr | [GMFSS_Fortuna](https://github.com/98mxr/GMFSS_Fortuna) 
| GIMM | GSeanCDAT | [GIMM](https://github.com/GSeanCDAT/GIMM-VFI) 
| IFRNet | ltkong218 | [IFRnet](https://github.com/ltkong218/IFRNet)

### Upscale Models:
| Model | Author | Link |
|--|--|--|
| 4x-SPANkendata | Crustaceous D | [4x-SPANkendata](https://openmodeldb.info/models/4x-SPANkendata) 
| 4x-ClearRealityV1 | Kim2091 | [4x-ClearRealityV1](https://openmodeldb.info/models/4x-ClearRealityV1) 
| 4x-Nomos8k-SPAN series | Helaman | [4x-Nomos8k-SPAN series](https://openmodeldb.info/models/4x-Nomos8k-span-otf-strong) 
| 2x-OpenProteus | SiroSky | [OpenProteus](https://github.com/Sirosky/Upscale-Hub/releases/tag/OpenProteus) 
| 2x-AnimeJaNai V2 and V3 Sharp | The Database | [AnimeJanai](https://github.com/the-database/mpv-upscale-2x_animejanai)
| 2x-AniSD | SiroSky | [AniSD](https://github.com/Sirosky/Upscale-Hub/releases/tag/AniSD)

### Decompression Models:
| Model | Author | Link |
|--|--|--|
| DeH264 | Helaman | [1xDeH264_realplksr](https://github.com/Phhofm/models/releases/tag/1xDeH264_realplksr) 


# Backends
  | Backend | Hardware | 
  |--|--|
  | TensorRT | NVIDIA RTX GPUs
  | PyTorch  | CUDA 12.6 and ROCm 6.2 capable GPUs
  | NCNN | Vulkan 1.3 capable GPUs
 
# FAQ
### General Application Usage
  | Question | Answer | 
  |--|--|
  | What does this program attempt to accomplish? | Fast, efficient and easily accessable video interpolation (Ex: 24->48FPS) and video upscaling (Ex: 1920->3840)
  | Why is it failing to recognize installed backends? | REAL Video Enhancer uses PIP and portable python for inference, this can sometimes have issues installing. Please attempt reinstalling the app before creating an issue.

### TensorRT related questions
  |||
  |--|--|
  | Why does it take so long to begin inference? | TensorRT uses advanced optimization at the beginning of inference based on your device, this is only done once per resolution of video inputed.
  | Why does the optimization and inference fail? | The most common way an optimization can fail is **Limited VRAM** There is no fix to this except using CUDA or NCNN instead.
 
### ROCm related questions
  |||
  |--|--|
  | Why am I getting (Insert Error here)? | ROCM is buggy, please take a look at <a href="https://github.com/TNTwise/REAL-Video-Enhancer/wiki/ROCm-Help">ROCm Help</a>.

### NCNN related questions
  |||
  |--|--|
  | Why am I getting (Insert Vulkan Error here)? | This usually is an OOM (Out Of Memory) error, this can indicate a weak iGPU or very old GPU, I recommeding trying out the <a href="https://github.com/TNTwise/REAL-Video-Enhancer-Colab">Colab Notebook</a>  instead.


# Cloning:
```
# Nightly
git clone --recurse-submodules https://github.com/TNTwise/REAL-Video-Enhancer 

# Stable
git clone --recurse-submodules https://github.com/TNTwise/REAL-Video-Enhancer --branch 2.3.8
```
# Building:

<p>3 supported build methods: </p>
<p> - pyinstaller (recommended for Win/Mac) <br/>
    - cx_freeze (recommended for Linux) <br/>
    - nuitka (experimental)
</p>
<p>supported python versions: </p>
<p> - 3.10 3.11, 3.12 <br/>
</p>

```
python3 build.py --build BUILD_OPTION --copy_backend
```

# Colab Notebook
 <a href="https://github.com/tntwise/REAL-Video-Enhancer-Colab">Colab Notebook</a>

# Credits:
### People:
| Person | For | Link |
|--|--|--|
| NevermindNilas | Some backend and reference code and working with me on many projects | https://github.com/NevermindNilas/ 
| Styler00dollar | RIFE ncnn models (4.1-4.5, 4.7-4.12-lite), Sudo Shuffle Span and benchmarking | https://github.com/styler00dollar 
| HolyWu | TensorRT engine generation code, inference optimizations, and RIFE jagged lines fixes | https://github.com/HolyWu/ 
| Rick Astley | Amazing music | https://www.youtube.com/watch?v=dQw4w9WgXcQ 

### Software: 
| Software Used | For | Link|
|--|--|--|
| FFmpeg | Multimedia framework for handling video, audio, and other media files | https://ffmpeg.org/ 
| QT | GUI framework | https://qt.io/
| FFMpeg Builds | Pre-compiled builds of FFMpeg. | Windows/Linux:  https://github.com/BtbN/FFmpeg-Builds, MacOS: https://github.com/eko5624/mpv-mac
| PyTorch | Neural Network Inference (CUDA/ROCm/TensorRT) | https://pytorch.org/ 
| NCNN | Neural Network Inference (Vulkan) | https://github.com/tencent/ncnn 
| RIFE | Real-Time Intermediate Flow Estimation for Video Frame Interpolation | https://github.com/hzwer/Practical-RIFE 
| rife-ncnn-vulkan | Video frame interpolation implementation using NCNN and Vulkan | https://github.com/nihui/rife-ncnn-vulkan 
| rife ncnn vulkan python | Python bindings for RIFE NCNN Vulkan implementation | https://github.com/media2x/rife-ncnn-vulkan-python 
| GMFSS | GMFlow based Anime VFI | https://github.com/98mxr/GMFSS_Fortuna
| GIMM | Motion Modeling Realistic VFI | https://github.com/GSeanCDAT/GIMM-VFI 
| ncnn python | Python bindings for NCNN Vulkan framework | https://pypi.org/project/ncnn 
| Real-ESRGAN | Upscaling | https://github.com/xinntao/Real-ESRGAN 
| SPAN | Upscaling | https://github.com/hongyuanyu/SPAN 
| Spandrel | CUDA upscaling model architecture support | https://github.com/chaiNNer-org/spandrel 
| ChaiNNer | Model Scale Detection | https://github.com/chaiNNer-org/chainner
| cx_Freeze | Tool for creating standalone executables from Python scripts (Linux build) | https://github.com/marcelotduarte/cx_Freeze 
| PyInstaller | Tool for creating standalone executables from Python scripts (Windows/Mac builds) | https://github.com/pyinstaller/pyinstaller
| Feather Icons | Open source icons library | https://github.com/feathericons/feather 
| PySceneDetect | Transition detection library for python | https://github.com/Breakthrough/PySceneDetect/
| Python Standalone Builds | Backend inference using portable python, helps when porting to different platforms. | https://github.com/indygreg/python-build-standalone |


# Star History
[![Star History Chart](https://api.star-history.com/svg?repos=tntwise/real-video-enhancer&type=Date)](https://star-history.com/#tntwise/real-video-enhancer&Date)
