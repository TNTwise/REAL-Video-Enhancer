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
Benchmarks done with 1920x1080 video, default settings.<br/>


| RX 6650 XT | Ensemble False | Ensemble True | 
|--|--|--|
| rife-v2.0 - v2.4  | 12.341 fps | -
| rife-v3.0 - v3.1 | 10.646 fps | -
| rife-v4.0 - v4.5 | 32.504 fps | -
| rife-v4.6 | 31.154 fps | 18.078 fps
| rife-v4.7 - v4.9 | 27.924 fps | 15.463 fps
| rife-v4.10 - v4.15 | 22.801 fps | 12.981 fps
| rife-v4.16-lite | 31.205 fps | 19.381 fps
## Benchmarks: (RIFE PyTorch CUDA)

| RTX 3080 | Ensemble False |
|--|--|
| rife-v4.6 | 81 fps 
| rife-v4.7 - v4.9 | 65 fps 
| rife-v4.10 - v4.15 | 55 fps 
| rife-v4.22-lite | 63 fps 

## Benchmarks: (RIFE PyTorch TensorRT)
| RTX 3080 | Ensemble False |
|--|--|
| rife-v4.6 | 220 fps 
| rife-v4.7 - v4.9 | 168 fps 
| rife-v4.10 - v4.15 | 133 fps 
| rife-v4.22-lite | 165 fps 

# Cloning:
```
git clone https://github.com/TNTwise/REAL-Video-Enhancer --branch main
```
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
  <li> <a rel="noopener noreferrer" href="https://github.com/styler00dollar" target="_blank">Styler00dollar (For RIFE models [4.1-4.5],[4.7-4.12-lite]) and Sudo Shuffle Span</a> </li>
  <li> <a rel="noopener noreferrer" href="https://github.com/nihui/ifrnet-ncnn-vulkan" target="_blank" >ifrnet-ncnn-vulkan</a>  </li>
  <li> <a rel="noopener noreferrer" href="https://github.com/hzwer/Practical-RIFE" target="_blank" >RIFE</a>  </li>
  <li> <a rel="noopener noreferrer" href="https://github.com/Breakthrough/PySceneDetect" target="_blank" >PySceneDetect</a>  </li>
  <li> <a rel="noopener noreferrer" href="https://github.com/nihui/realcugan-ncnn-vulkan" target="_blank" >realcugan-ncnn-vulkan</a>  </li>
  <li> <a rel="noopener noreferrer" href="https://github.com/bilibili/ailab/tree/main/Real-CUGAN" target="_blank" >REAL-Cugan</a>  </li>
  <li> <a rel="noopener noreferrer" href="https://github.com/NevermindNilas/TheAnimeScripter" target="_blank" >TheAnimeScripter (For CUDA implementation code, TRT upscaling code and non image extraction NCNN code in v1)</a></li>
  <li> <a rel="noopener noreferrer" href="https://github.com/chaiNNer-org/spandrel" target="_blank">Spandrel (For CUDA upscaling model arch support)</a></li>
  <li> <a rel="noopener noreferrer" href="https://github.com/Final2x/realesrgan-ncnn-py" target="_blank">RealESRGAN NCNN python</a></li>
  <li> <a rel="noopener noreferrer" href="https://github.com/marcelotduarte/cx_Freeze" target="_blank">cx_Freeze</a></li>
  <li> <a rel="noopener noreferrer" href="https://github.com/WolframRhodium" target="_blank">WolframRhodium for rife v2 code.</a></li>
  <li> <a rel="noopener noreferrer" href="https://github.com/elexor" target="_blank">elexor for porting rife v2 to older versions of rife</a></li>
  <li> <a rel="noopener noreferrer" href="https://github.com/HolyWu/vs-rife" target="_blank">vs-rife (For rife TRT inference code)</a></li>
        
  <li> <a rel="noopener noreferrer" href="https://github.com/feathericons/feather" target="_blank">feather icons</a></li>
</ul>


# Custom models used:

<ul>
  <li> <a rel="noopener noreferrer" href="https://openmodeldb.info/models/4x-SPANkendata" target="_blank" >4x-SPANkendata by Crustaceous D</a> </li>
  <li> <a rel="noopener noreferrer" href="https://openmodeldb.info/models/4x-ClearRealityV1" target="_blank" >4x-ClearRealityV1 by Kim2091</a> </li>
  <li> <a rel="noopener noreferrer" href="https://openmodeldb.info/models/4x-Nomos8k-span-otf-strong" target="_blank" >4x-Nomos8k-SPAN series by Helaman</a> </li>
  <li> <a rel="noopener noreferrer" href="https://github.com/Sirosky/Upscale-Hub/releases/tag/OpenProteus" target="_blank" >OpenProteus by SiroSky</a> </li>
</ul>


