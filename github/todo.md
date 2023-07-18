
Todo:<br/>
Finish get models function<br/>
Output video extension as input vidoe, fix audio by copying audio from input video to output video as output video processces, or by encoding audio both times (prob shouldnt tbh)<br/>
fix stupid issue where inputing a video from the render directory wont work(im stupid)<br/>
Add frames to new video output as rendered, and delete rendered frames???(might be a feature in ffmpeg???)<br/>
split script up by removing circular imports, and importing maineindow from main script, this alloes me to control everythign i need to in different ecripts. makes it more clean and easy to manage when different parts of gui code are easily accessabe and readable<br/>
maybe use a function ina different script to return progressbar percentage from specific values, could prove better than current implementation 

IMPORTANT<br/>

Implement a system where if a setting does not exist, append that setting with its default value to the settings file instead of resetting the entire settings file<br />
add vs rife support as it implements better with app<br/>

var cmd = `${inject_env} && "${vspipe}" --arg "tmp=${path.join(cache, "tmp.json")}" -c y4m "${engine}" - -p | "${ffmpeg}" -y -loglevel error -i pipe: ${params} "${tmpOutPath}"`; 

this should help with rife vs support<br/>

add discord rpc support
