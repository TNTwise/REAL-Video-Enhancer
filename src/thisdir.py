import os
homedir =  os.path.expanduser(r"~")
if os.path.exists(f'{homedir}/.var/app/io.github.tntwise.REAL-Video-Enhancer') == False:
    os.system(f'mkdir -p "{homedir}/.var/app/io.github.tntwise.REAL-Video-Enhancer"')
def thisdir():
    return (f'{homedir}/.var/app/io.github.tntwise.REAL-Video-Enhancer')