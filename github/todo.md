
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
Add indicator for file drag and drop<br/>
implement warnings for space with realesrgan<br/>
clean up settings<br/>
add more AI options<br/>
implement storage optimized option fully. (including transition detection)<br/>
add tab for upscaling images and not just videos<br/>
add different themes<br/>
add system where you can pause a render, have it save where it took place to a file, and can read that file to resume that render.<br/>
notification to say when render is finished<br/>
add automatic detection for how many frames should be rendered<br/>
generate script based on module for AI(so its easier to implement them)<br/>
fix issue where sometimes realesrgan times is disabled, even on animation(happens for waifu2x too)<br/>
automatically download models that dont exist
