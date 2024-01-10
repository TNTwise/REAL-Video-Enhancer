 
import os
import subprocess
import re
import math
def extract(Image_Type,input_file,SceneChangeDetection,times,amount_of_zeros=8):
    os.chdir(os.path.dirname(file))
    os.system('rm -rf transitions')
    os.mkdir('transitions/') 
    if Image_Type != '.webp':
                    ffmpeg_cmd = f'ffmpeg -i "{input_file}" -filter_complex "select=\'gt(scene\,{SceneChangeDetection})\',metadata=print" -vsync vfr -q:v 1 "transitions/%07d.{Image_Type}"' 
    else:
                    ffmpeg_cmd = f'ffmpeg -i "{input_file}" -filter_complex "select=\'gt(scene\,{SceneChangeDetection})\',metadata=print" -vsync vfr -q:v 100 "transitions/%07d.png"'

    output = subprocess.check_output(ffmpeg_cmd, shell=True, stderr=subprocess.STDOUT)
    
    output_lines = output.decode("utf-8").split("\n")
    timestamps = []

    
    ffprobe_cmd = f"ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate {input_file}"
    result = subprocess.check_output(ffprobe_cmd, shell=True).decode("utf-8")

    match = re.match(r'(\d+)/(\d+)', result)
    
    numerator, denominator = match.groups()
    fps = int(numerator) / int(denominator)
        

    fps_value = int(numerator) / int(denominator) if match else None
    for line in output_lines:
                if "pts_time" in line:
                    timestamp = str(line.split("_")[3])
                    timestamp = str(timestamp.split(':')[1])
                    timestamps.append(math.ceil(round(float(timestamp)*float(fps_value))*times))
    transitions = os.listdir('transitions/')
    for iteration,i in enumerate(transitions):
        
        if Image_Type != '.webp':
                    os.system(f'mv transitions/{str(str(iteration+1).zfill(7))}.{Image_Type} transitions/{timestamps[iteration]}.{Image_Type}')
        else:
                    os.system(f'mv transitions/{str(str(iteration+1).zfill(7))}.png transitions/{timestamps[iteration]}.{Image_Type}')
    for i in timestamps:
            for j in range(math.ceil(times)):
                    os.system(f'cp transitions/{i}.{Image_Type} transitions/{str(int(i)-j).zfill(amount_of_zeros)}.{Image_Type}' )
            os.remove(f'transitions/{i}.{Image_Type}')
image = input('Please pick an image type:\n1:PNG\n2:JPG\n3:WEBP\n(Please pick 1,2 or 3): ')
if image == '1':
        image = 'png'
elif image == '2':
        image = 'jpg'
elif image == '3':
        image = 'webp'
else:
    print('Invalid answer!')
    exit()

file = input('\nPlease paste input file location here: ')
if os.path.isfile(file) == False:
        print('Not a file!')
        exit()

sensativity=input('\nPlease enter the sensitivity (0-9)\n(0 means most sensitive, meaning it will detect the most frames as scene changes)\n(9 being the least sensitive, meaning it will detect the least amount of frames as scene changes.)\n: ')
try:
    if int(sensativity) > 9 or int(sensativity) < 0:
            print('invalid sensitivity')
            exit()
except:
        print('Not an integer!')
        exit()
try:
    timestep = input('\nPlease enter a timestep (not the number of frames, just the multiplier): ')
except:
        print('Not a float!')
        exit()
try:
    if len(timestep) > 1:
        timestep = float(timestep)
    else:
        timestep=int(timestep)
except:
        print('invalid!')
        exit()
try: 
    amount_of_zeros = int(input('\nPlease enter the num you put in for the amount of zeros per frame, default = %08d\nif you changed this value in extraction of frames, please put that value here\nif not, just skip this. '))
except:
        amount_of_zeros=8
extract(image,file,f'0.{sensativity}',timestep,amount_of_zeros)