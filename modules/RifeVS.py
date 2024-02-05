import sys
import os
import vapoursynth as vs
from vapoursynth import core
import src.programData.thisdir
thisdir = src.programData.thisdir.thisdir()
core = vs.core
core.num_threads = 8  # can influence ram usage
core.std.LoadPlugin(path="/home/pax/VapourSynth-RIFE-ncnn-Vulkan/librife.so")
clip = core.bs.VideoSource(source="MFGhost-OP1.webm")
clip = core.resize.Bilinear(clip, format=vs.RGBS, matrix_in_s="709")

# detects scene changes, requires misc plugin
#clip = core.misc.SCDetect(clip)

clip = core.rife.RIFE(
    
    clip,
    
    model=43,
    gpu_thread=5,
    factor_num=4,
    factor_den=2,
    gpu_id=0,
    
    tta=False,
    uhd=False,
    skip=True,
    sc=False,

    
    
)

clip = vs.core.resize.Bilinear(clip, format=vs.YUV420P8, matrix_s="709")
clip.set_output(index=0)