# REAL Video Enhancer
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FTNTwise%2FREAL-Video-enhancer%2F&countColor=%23263759)
![downloads_total](https://img.shields.io/github/downloads/tntwise/REAL-Video-Enhancer/total.svg?label=downloads%40total)
[![pypresence](https://img.shields.io/badge/using-pypresence-00bb88.svg?style=for-the-badge&logo=discord&logoWidth=20)](https://github.com/qwertyquerty/pypresence)
<a href="https://discord.gg/hwGHXga8ck">
      <img src="https://img.shields.io/discord/1041502781808328704?label=Discord" alt="Discord Shield"/></a>

### Now out on flathub!
<a href="https://flathub.org/apps/io.github.tntwise.REAL-Video-Enhancer">https://flathub.org/apps/io.github.tntwise.REAL-Video-Enhancer</a>
<p align=center>
  <img src="https://github.com/TNTwise/REAL-Video-Enhancer/raw/2.0/icons/logo-v2.svg" width = "25%">
</p>

<strong>REAL Video Enhancer</strong>  is a redesigned and enhanced version of the original Rife ESRGAN App for Linux. This program offers convenient access to frame interpolation and upscaling functionalities on Linux, and is an alternative to outdated software like <a rel="noopener noreferrer" href="https://nmkd.itch.io/flowframes" target="_blank" >Flowframes</a> or <a rel="noopener noreferrer" href="https://github.com/mafiosnik777/enhancr" target="_blank">enhancr</a> on Windows.

V2 Alpha 2 New Look!:
<p align=center>
  <img src="https://github.com/TNTwise/REAL-Video-Enhancer/blob/2.0/icons/demo.png" width = "100%">
</p>
<h1>Features: </h1>
<ul>
  <li> <strong>NEW!</strong> Windows support. </li> 
  <li> CUDA support. </li>
  <li> MacOS support. </li>
  <li>Support for Ubuntu 20.04+ on AppImage and Flatpak. </li>
  <li> Discord RPC support for Discord system package and Discord flatpak. </li>
  <li> Scene change detection to preserve sharp transitions. </li>
  <li> Preview that shows latest frame that has been rendered. </li>
</ul>

## Benchmarks: (RIFE NCNN)
Benchmarks done with 1920x1080 video, default settings.<br/>


| RX 6650 XT | |
 |--|--|
| rife-v4.6 | 31 fps 
| rife-v4.7 - v4.9 | 28 fps
| rife-v4.10 - v4.15 | 23 fps 
| rife-v4.16-lite | 31 fps 

| RTX 3080 | |
|--|--|
| rife-v4.6 | 81 fps 
| rife-v4.7 - v4.9 | 65 fps 
| rife-v4.10 - v4.15 | 55 fps 
| rife-v4.22 | 50 fps 
| rife-v4.22-lite | 63 fps 

## Benchmarks: (RIFE TensorRT 10.3)
| RTX 3080 | |
|--|--|
| rife-v4.6 | 270 fps 
| rife-v4.7 - v4.9 | 204 fps 
| rife-v4.10 - v4.15 | 166 fps 
| rife-v4.22 | 140 fps 
| rife-v4.22-lite | 192 fps 

# Cloning:
```
git clone https://github.com/TNTwise/REAL-Video-Enhancer
```
# Building:
```
python3 build.py --build_exe
```
## Download the Latest PRE-Release release here:
<strong> </strong> <a href="https://github.com/TNTwise/REAL-Video-Enhancer/releases/tag/prerelease">https://github.com/TNTwise/REAL-Video-Enhancer/releases/tag/prerelease</a>

# Software used:

<ul>
  <li> <a rel="noopener noreferrer" href="https://ffmpeg.org/" target="_blank" >FFMpeg</a> </li>
  <li> <a rel="noopener noreferrer" href="https://github.com/nihui/rife-ncnn-vulkan" target="_blank" >rife-ncnn-vulkan</a> </li>
  <li> <a rel="noopener noreferrer" href="https://github.com/xinntao/Real-ESRGAN" target="_blank" >Real-ESRGAN</a> </li>
  <li> <a rel="noopener noreferrer" href="https://github.com/styler00dollar" target="_blank">Styler00dollar (For RIFE models [4.1-4.5],[4.7-4.12-lite]) and Sudo Shuffle Span</a> </li>
  <li> <a rel="noopener noreferrer" href="https://github.com/hzwer/Practical-RIFE" target="_blank" >RIFE</a>  </li>
  <li> <a rel="noopener noreferrer" href="https://github.com/Breakthrough/PySceneDetect" target="_blank" >PySceneDetect</a>  </li>
  <li> <a rel="noopener noreferrer" href="https://github.com/NevermindNilas/TheAnimeScripter" target="_blank" >TheAnimeScripter for inspiration and mods to rife arch.</a></li>
  <li> <a rel="noopener noreferrer" href="https://github.com/chaiNNer-org/spandrel" target="_blank">Spandrel (For CUDA upscaling model arch support)</a></li>
  <li> <a rel="noopener noreferrer" href="https://github.com/Final2x/realesrgan-ncnn-py" target="_blank">RealESRGAN NCNN python</a></li>
  <li> <a rel="noopener noreferrer" href="https://github.com/marcelotduarte/cx_Freeze" target="_blank">cx_Freeze</a></li>
  <li> <a rel="noopener noreferrer" href="https://github.com/media2x/rife-ncnn-vulkan-python" target="_blank">rife ncnn vulkan python</a></li>
  <li> <a rel="noopener noreferrer" href="https://github.com/hongyuanyu/SPAN" target="_blank">SPAN</a></li>
  <li> <a rel="noopener noreferrer" href="https://github.com/WolframRhodium" target="_blank">WolframRhodium for rife v2 code.</a></li>
  <li> <a rel="noopener noreferrer" href="https://github.com/elexor" target="_blank">elexor for porting rife v2 to older versions of rife</a></li>
  <li> <a rel="noopener noreferrer" href="https://github.com/HolyWu/vs-rife" target="_blank">vs-rife (For TRT engine generation code)</a></li>
        
  <li> <a rel="noopener noreferrer" href="https://github.com/feathericons/feather" target="_blank">feather icons</a></li>
</ul>


# Custom models used:

<ul>
  <li> <a rel="noopener noreferrer" href="https://openmodeldb.info/models/4x-SPANkendata" target="_blank" >4x-SPANkendata by Crustaceous D</a> </li>
  <li> <a rel="noopener noreferrer" href="https://openmodeldb.info/models/4x-ClearRealityV1" target="_blank" >4x-ClearRealityV1 by Kim2091</a> </li>
  <li> <a rel="noopener noreferrer" href="https://openmodeldb.info/models/4x-Nomos8k-span-otf-strong" target="_blank" >4x-Nomos8k-SPAN series by Helaman</a> </li>
  <li> <a rel="noopener noreferrer" href="https://github.com/Sirosky/Upscale-Hub/releases/tag/OpenProteus" target="_blank" >OpenProteus by SiroSky</a> </li>
</ul>


