# REAL Video Enhancer
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FTNTwise%2FREAL-Video-enhancer%2F&countColor=%23263759)
![downloads_total](https://img.shields.io/github/downloads/tntwise/REAL-Video-Enhancer/total.svg?label=downloads%40total)
[![pypresence](https://img.shields.io/badge/using-pypresence-00bb88.svg?style=for-the-badge&logo=discord&logoWidth=20)](https://github.com/qwertyquerty/pypresence)
<a href="https://discord.gg/hwGHXga8ck">
      <img src="https://img.shields.io/discord/1041502781808328704?label=Discord" alt="Discord Shield"/></a>

### Now out on flathub!
<a href="https://flathub.org/apps/io.github.tntwise.REAL-Video-Enhancer">https://flathub.org/apps/io.github.tntwise.REAL-Video-Enhancer</a>
<p align=center>
  <img src="https://github.com/TNTwise/REAL-Video-Enhancer/blob/main/icons/logo%20v1.png" width = "25%">
</p>

<strong>REAL Video Enhancer</strong>  is a redesigned and enhanced version of the original Rife ESRGAN App for Linux. This program offers convenient access to frame interpolation and upscaling functionalities on Linux, and is an alternative to outdated software like <a rel="noopener noreferrer" href="https://nmkd.itch.io/flowframes" target="_blank" >Flowframes</a> or <a rel="noopener noreferrer" href="https://github.com/mafiosnik777/enhancr" target="_blank">enhancr</a> on Windows.
<p align=center>
  <img src="https://github.com/TNTwise/REAL-Video-Enhancer/blob/main/Screenshots/mainmenu.png" width = "100%">
</p>
<p align=center>
  <img src="https://github.com/TNTwise/REAL-Video-Enhancer/blob/main/Screenshots/settings.png" width = "100%">
</p>
<h1>Features: </h1>
<ul>
  <li> <strong>NEW!</strong> CUDA support. </li>
  <li> <strong>NEW!</strong> MacOS support. </li>
  <li>Support for Ubuntu 20.04+ on AppImage and Flatpak. </li>
  <li> Discord RPC support for Discord system package and Discord flatpak. </li>
  <li> Scene change detection to preserve sharp transitions. </li>
  <li> Preview that shows latest frame that has been rendered. </li>
  <li> Enhancing straight from a YouTube link or a video URL.  </li>
</ul>

## Benchmarks: (RIFE NCNN)
Benchmarks done with 1920x1080 video, default settings using JPG image extraction.<br/>


| RX 6650 XT | Ensemble False | Ensemble True | 
|--|--|--|
| rife-v2.0 - v2.4  | 12.341 fps | -
| rife-v3.0 - v3.1 | 10.646 fps | -
| rife-v4.0 - v4.5 | 32.504 fps | -
| rife-v4.6 | 31.154 fps | 18.078 fps
| rife-v4.7 - v4.9 | 27.924 fps | 15.463 fps
| rife-v4.10 - v4.15 | 22.801 fps | 12.981 fps
| rife-v4.16-lite | 31.205 fps | 19.381 fps

# Building:
```
python3 build.py
```
## Download the Latest PRE-Release release here:
<strong> </strong> <a href="https://github.com/TNTwise/REAL-Video-Enhancer/releases/tag/prerelease">https://github.com/TNTwise/REAL-Video-Enhancer/releases/tag/prerelease</a>

# Software used:

<ul>
  <li> <a rel="noopener noreferrer" href="https://ffmpeg.org/" target="_blank" >FFMpeg</a> </li>
  <li> <a rel="noopener noreferrer" href="https://github.com/nihui/rife-ncnn-vulkan" target="_blank" >rife-ncnn-vulkan</a> </li>
  <li> <a rel="noopener noreferrer" href="https://github.com/xinntao/Real-ESRGAN" target="_blank" >Real-ESRGAN</a> </li>
  <li> <a rel="noopener noreferrer" href="https://github.com/yt-dlp/yt-dlp" target="_blank" >yt-dlp</a>  </li>
  <li> <a rel="noopener noreferrer" href="https://github.com/styler00dollar/VapourSynth-RIFE-ncnn-Vulkan" target="_blank">VapourSynth-RIFE-ncnn-Vulkan (For RIFE models [4.1-4.5],[4.7-4.12-lite])</a> </li>
  <li> <a rel="noopener noreferrer" href="https://github.com/nihui/ifrnet-ncnn-vulkan" target="_blank" >ifrnet-ncnn-vulkan</a>  </li>
  <li> <a rel="noopener noreferrer" href="https://github.com/hzwer/Practical-RIFE" target="_blank" >RIFE</a>  </li>
  <li> <a rel="noopener noreferrer" href="https://github.com/Breakthrough/PySceneDetect" target="_blank" >PySceneDetect</a>  </li>
  <li> <a rel="noopener noreferrer" href="https://github.com/nihui/realcugan-ncnn-vulkan" target="_blank" >realcugan-ncnn-vulkan</a>  </li>
  <li> <a rel="noopener noreferrer" href="https://github.com/bilibili/ailab/tree/main/Real-CUGAN" target="_blank" >REAL-Cugan</a>  </li>
  <li> <a rel="noopener noreferrer" href="https://github.com/NevermindNilas/TheAnimeScripter" target="_blank" >TheAnimeScripter (For CUDA implementation code, TRT upscaling code and non image extraction NCNN code in v1)</a></li>
  <li> <a rel="noopener noreferrer" href="https://github.com/chaiNNer-org/spandrel" target="_blank">Spandrel (For CUDA upscaling model arch support)</a></li>
  <li> <a rel="noopener noreferrer" href="https://github.com/Final2x/realesrgan-ncnn-py" target="_blank">RealESRGAN NCNN python</a></li>
  <li> <a rel="noopener noreferrer" href="https://github.com/marcelotduarte/cx_Freeze" target="_blank">cx_Freeze</a></li>
  <li> <a rel="noopener noreferrer" href="https://github.com/chaiNNer-org/chaiNNer/blob/2aa0b46233ba8cd90d4bb405e2bc6e16a3430546/backend/src/nodes/impl/ncnn/model.py
  " target="_blank">Chainner's NCNN Implementation (For scale detection with NCNN models)
  <li> <a rel="noopener noreferrer" href="https://github.com/HolyWu/vs-rife" target="_blank">vs-rife (For rife TRT inference code)</a></li>
</ul>

# Custom models used:

<ul>
  <li> <a rel="noopener noreferrer" href="https://openmodeldb.info/models/4x-SPANkendata" target="_blank" >4x-SPANkendata by Crustaceous D</a> </li>
  <li> <a rel="noopener noreferrer" href="https://openmodeldb.info/models/4x-ClearRealityV1" target="_blank" >4x-ClearRealityV1 by Kim2091</a> </li>
  
  
</ul>


