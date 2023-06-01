#damn it really took like 5 hours for me to get this shit
# command to run vspipe -c y4m rifeVS.py - | x264 --demuxer y4m - --output encoded.mp4

import vapoursynth as vs
import ffms2
core = vs.core
mp4_file=''
plugin = core.std.LoadPlugin("/home/pax/Downloads/vsRife/vs-rife/librife.so")
clip = core.lsmas.LWLibavSource(source=f"{mp4_file}", cache=0)
clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")
clip = core.rife.RIFE(clip)
clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")

clip.set_output()
