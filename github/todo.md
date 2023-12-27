
Todo:<br/>
Finish get models function<br/>
Output video extension as input vidoe, fix audio by copying audio from input video to output video as output video processces, or by encoding audio both times (prob shouldnt tbh)<br/>
Add frames to new video output as rendered, and delete rendered frames???(might be a feature in ffmpeg???)<br/>

IMPORTANT<br/>

add vs rife support as it implements better with app<br/>

var cmd = `${inject_env} && "${vspipe}" --arg "tmp=${path.join(cache, "tmp.json")}" -c y4m "${engine}" - -p | "${ffmpeg}" -y -loglevel error -i pipe: ${params} "${tmpOutPath}"`; 

this should help with rife vs support<br/>

implement modular system where app can import modules from scripts(either iterate through module directory or add specific button in settings)<br/>
<br/>
add more AI options<br/>
add different themes<br/>
add system where you can pause a render, have it save where it took place to a file, and can read that file to resume that render.<br/>
generate script based on module for AI(so its easier to implement them)<br/>
fix issue where sometimes realesrgan times is disabled, even on animation(happens for waifu2x too)<br/>
add better downloads that dont rely on github bin dir??, sometimes slow<br/>
run error pop up on original thread<br/>
Add pausing by rendering current frames rendered in optimized preset into a video, and then concating all videos together, and removing all of input_frames / self.times frames. Then re-start rife in the same folder to continue<br/>
add checks periodically in the render to check amount of frames remaining, and if ffmpeg is playing catchup, kill render and render everything out.<br/>
add ifrnet-ncnn-vulkan(seems promising)<br/>
use iteration to iterate through every image and render them individually, may help with pausing(might use rife ncnn vulkan python package to achieve this cleanly?)<br/>
list amount of transitions detected in logs<br/>
fix randomly graying out rife times?<br/>
choose install location<br/>
make full queueing system<br/>
make install/remove models page add custom models, so they wont get deleted on install/remove(iterate through directory and add add checkbox per custom model?)<br/>
setting of export video to same directory as input video by defualt<br/>
add guide on how to add/convert custom models<br/>
add uhd/tta options<br/>
finish esrgan custom model support by asking what resolution increase it will be per model added.<br/>
fix random crashing after inputing non youtube url, just implement proper threading.<br/>
fix popup for resolution showing up one time after a video that would call it is selected.<br/>
detect when a esrgan model isnt converted correctly (didnt change data to input)<br/>
add option for ffmpeg speed/quality preset<br/>
enter gpu id via text(like upscayl)<br/>
add option for after render is done, like shutdown.<br/>
Fix realesrgan not getting resolution from linked videos<br/>
put render directories into a single folder for easier cleaning<br/>
make transition detection dynamic between fps, like it can be 2x, 2.5x, 2.7x, any custom timestep for the rife-v4 models<br/>
add documentation<br/>
repeating logs, if an error has been repeated, dont log it a bunch of times, instead log it once and say how many times it failed.<br/>
select which log file to view<br/>
add dependencies install script for ytdlp, so it can automatically be updated.
Wiki page<br/>
create a python interface to easily interact with ncnn, more and create the optimized method, and also use it to add vapoursynth support.
 - How to Uscale<br/>
 - Best uscaling practices<br/>
 - Explanation of Settings<br/>
 - Proper guide to import ESRGAN models<br/>
 - Explanation behind different parts of the app<br/>